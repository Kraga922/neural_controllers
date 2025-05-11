import os
import pickle
import pandas as pd
import re
from glob import glob

import os

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']
RESULTS_DIR = os.path.join(NEURAL_CONTROLLERS_DIR, 'results')

LABELS = {
    'None' : 'ToxicChat-T5-Large',
    'gpt-4o' : 'GPT-4o',
    'llama_3.1_70b_4bit_it' : 'Llama 3.1 70b Instruct (4-bit)',
    'llama_3.3_70b_4bit_it' : 'Llama 3.3 70b Instruct (4-bit)',
    'llama_3_8b_it' : 'Llama 3.1 8b Instruct',
    'logistic' : 'Logistic',
    'linear' : 'Lin. Reg.',
    'rfm' : 'RFM',
}

DATASETS = {
    'toxic_chat' : 'ToxicChat',
    'fava' : 'FAVA',
    'halu_eval_general' : 'HaluEval (General)',
    'halu_eval_wild' : 'HaluEval (Wild)',
    'pubmedQA' : 'PubMedQA',
    'RAGTruth' : 'RAGTruth',
}

JUDGE_METHODS = set(['GPT-4o', 'Llama 3.1 8b Instruct', 'ToxicChat-T5-Large', 'Llama 3.1 70b Instruct (4-bit)', 'Llama 3.3 70b Instruct (4-bit)'])


# Recursively find all *_metrics.pkl files in results
metrics_files = [y for x in os.walk(RESULTS_DIR) for y in glob(os.path.join(x[0], '*metrics.pkl'))]
print(f"Found {len(metrics_files)} metrics files in {RESULTS_DIR}")
rows = []

# Regex to parse filenames for new formats
probe_re = re.compile(r'(?P<dataset>[^-]+)-(?P<model>[^-]+)-(?P<method>[^-]+)-prompt_(?P<prompt_version>v\d+)-tuning_metric_(?P<tuning_metric>[^-]+)-top_k_(?P<n_components>\d+)-(?P<agg_type>aggregated|best_layer)_metrics\.pkl$')
judge_re = re.compile(r'^(?P<dataset>.+?)-(?P<judge_type>.+?)-(?P<judge_model>.+?)-prompt_(?P<prompt_version>v\d+)-metrics\.pkl$')

for file in metrics_files:
    try:
        with open(file, 'rb') as f:
            metrics = pickle.load(f)
    except Exception as e:
        print(f"Could not load {file}: {e}")
        continue
    fname = os.path.basename(file)
    # Try to match probe or judge pattern
    m_probe = probe_re.match(fname)
    m_judge = judge_re.match(fname)
    if m_probe:
        # print(f"Found probe file: {fname}")
        row = {
            'file': file,
            'dataset': m_probe.group('dataset'),
            'model': m_probe.group('model'),
            'method': m_probe.group('method'),
            'prompt_version': m_probe.group('prompt_version'),
            'tuning_metric': m_probe.group('tuning_metric'),
            'n_components': m_probe.group('n_components'),
            'aggregation': m_probe.group('agg_type'),
        }
    elif m_judge:
        print(f"Found judge file: {fname}")
        row = {
            'file': file,
            'dataset': m_judge.group('dataset'),
            'judge_type': m_judge.group('judge_type'),
            'judge_model': m_judge.group('judge_model'),
            'prompt_version': m_judge.group('prompt_version'),
            'aggregation': '',
        }
    else:
        # If filename doesn't match either format, print for debugging and skip this file
        # if 'openai' in fname or 'llama_llama' in fname:
        #     print(f"Filename did not match probe or judge regex: {fname}")
        continue
    # Flatten metrics dict for DataFrame
    for k, v in metrics.items():
        row[k] = v
    rows.append(row)

# Create DataFrame
if rows:
    df = pd.DataFrame(rows)

    # Reorder columns
    cols = [
        'dataset', 'model', 'method', 'judge_type', 'judge_model',
        'prompt_version', 'tuning_metric', 'n_components', 'aggregation', 'auc', 'file'
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    # Print table
    # print(df.to_string(index=False))
    # Save as CSV
    out_csv = os.path.join(RESULTS_DIR, 'all_results_table.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved table to {out_csv}")

    # Prepare for LaTeX output: only dataset, method, auc
    df_latex = df.copy()
    def format_method(row):
        # If judge_model is present and non-empty, use its label
        if pd.notnull(row.get('judge_model', None)) and row.get('judge_model', '') != '':
            judge_label = LABELS.get(row['judge_model'], row['judge_model'])
            return judge_label
        # Otherwise, use method label and append aggregation if present
        method_label = LABELS.get(row.get('method', ''), row.get('method', ''))
        agg = row.get('aggregation', '')
        if agg:
            return f"{method_label} ({agg})"
        return method_label
    df_latex['method'] = df_latex.apply(format_method, axis=1)
    # Only keep relevant columns
    latex_cols = [c for c in ['dataset', 'method', 'auc'] if c in df_latex.columns]
    df_latex = df_latex[latex_cols]

    # Pivot so each dataset is a column, each row is a method, auc is the value
    df_pivot = df_latex.pivot_table(index='method', columns='dataset', values='auc')

    # Reorder rows: RFM first, judge models last
    method_names = df_pivot.index.tolist()
    # Find which method names correspond to judge models in the original df_latex
    judge_rows = [m for m in method_names if m in JUDGE_METHODS]
    
    # RFM row (may have aggregation tag)
    rfm_rows = [m for m in method_names if m.startswith('RFM')]
    # All other rows
    other_rows = [m for m in method_names if m not in rfm_rows and m not in judge_rows]
    # New order
    new_order = rfm_rows + other_rows + judge_rows
    df_pivot = df_pivot.loc[new_order]

    # Truncate to three decimals for LaTeX output (not round)
    def truncate(x):
        if pd.isna(x):
            return '-'
        if isinstance(x, float):
            return f"{int(x * 1000) / 1000:.3f}"
        return x
    df_pivot_fmt = df_pivot.applymap(truncate)

    # Add \textbf{} to max value in each column
    for col in df_pivot.columns:
        # Get column as float for comparison, ignore non-float entries
        col_vals = df_pivot[col]
        # Mask for valid floats
        valid_mask = col_vals.apply(lambda x: isinstance(x, float) and not pd.isna(x))
        if valid_mask.any():
            max_val = col_vals[valid_mask].max()
            max_str = f"{int(max_val * 1000) / 1000:.3f}"
            # Bold all matches to max_str in the formatted DataFrame
            df_pivot_fmt[col] = df_pivot_fmt[col].apply(lambda x: f"\\textbf{{{x}}}" if x == max_str else x)

    # Map dataset column names to labels and order columns
    dataset_order = ['fava', 'halu_eval_general', 'halu_eval_wild', 'pubmedQA', 'RAGTruth', 'toxic_chat']
    dataset_labels = [DATASETS.get(ds, ds) for ds in dataset_order if ds in df_pivot.columns]
    dataset_col_map = {ds: DATASETS.get(ds, ds) for ds in df_pivot.columns}

    # Reorder and relabel columns for main table
    ordered_cols = [ds for ds in dataset_order if ds in df_pivot.columns]
    df_pivot_fmt = df_pivot_fmt[ordered_cols]
    df_pivot_fmt.columns = [DATASETS.get(ds, ds) for ds in ordered_cols]

    print("\nLaTeX table (methods as rows, datasets as columns, auc as value):")
    print(df_pivot_fmt.to_latex(index=True, na_rep='-'))

    # --- Best Probing Method vs Judge Models Table ---
    # Get best probing method (excluding judge models)
    probe_methods = [m for m in method_names if m not in JUDGE_METHODS]
    df_probe = df_pivot.loc[probe_methods, ordered_cols]
    best_auc_probe = df_probe.apply(lambda col: col.max(), axis=0)
    
    # Get all judge models
    df_judges = df_pivot.loc[judge_rows, ordered_cols]
    
    # Combine into single comparison table
    comparison_df = pd.DataFrame(best_auc_probe).T
    comparison_df = pd.concat([comparison_df, df_judges])
    comparison_df.index = ['Best Probing Method'] + judge_rows
    comparison_df.columns = [DATASETS.get(ds, ds) for ds in ordered_cols]

    # Format as before (truncate, bold)
    def format_best(x, col, df):
        if pd.isna(x):
            return '-'
        if isinstance(x, float):
            val_str = f"{int(x * 1000) / 1000:.3f}"
            # Bold if it's the max in this column
            if x == df[col].max():
                return f"\\textbf{{{val_str}}}"
            return val_str
        return x
    
    comparison_df_fmt = comparison_df.copy()
    for col in comparison_df.columns:
        comparison_df_fmt[col] = comparison_df[col].apply(lambda x: format_best(x, col, comparison_df))

    print("\nLaTeX table (Best Probing Method vs Judge Models):")
    print(comparison_df_fmt.to_latex(index=True, na_rep='-'))
else:
    print("No metrics files found.")
