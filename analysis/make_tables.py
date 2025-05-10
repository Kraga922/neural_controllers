import os
import pickle
import pandas as pd
import re
from glob import glob

import os

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']
RESULTS_DIR = os.path.join(NEURAL_CONTROLLERS_DIR, 'results')

# Recursively find all *_metrics.pkl files in results
metrics_files = [y for x in os.walk(RESULTS_DIR) for y in glob(os.path.join(x[0], '*_metrics.pkl'))]
print(f"Found {len(metrics_files)} metrics files in {RESULTS_DIR}")
rows = []

# Regex to parse filenames
filename_re = re.compile(r'(?P<method>[\w\-]+)_(?P<judge_model>[\w\-]+)?(?:_(?P<task>[\w\-]+))?(?:_prompt(?:_version)?_?(?P<prompt_version>v\d+))?(?:_(?P<agg_type>aggregated|best_layer))?_metrics.pkl$')

for file in metrics_files:
    try:
        with open(file, 'rb') as f:
            metrics = pickle.load(f)
    except Exception as e:
        print(f"Could not load {file}: {e}")
        continue
    # Try to parse method, judge_model, task, agg_type from filename
    fname = os.path.basename(file)
    m = filename_re.search(fname)
    if m:
        method = m.group('method') or ''
        judge_model = m.group('judge_model') or ''
        task = m.group('task') or ''
        prompt_version = m.group('prompt_version') or ''
        agg_type = m.group('agg_type') or ''
    else:
        method = fname
        judge_model = ''
        task = ''
        prompt_version = ''
        agg_type = ''
    # Infer dataset/task from directory structure
    rel_path = os.path.relpath(file, RESULTS_DIR)
    parts = rel_path.split(os.sep)
    # If in halubench_results/<type>/..., use <type> as task
    if len(parts) >= 3 and parts[0] == 'halubench_results':
        task = parts[1]
    # Otherwise, use first-level subdir if task not already set
    elif not task and len(parts) >= 2:
        if parts[0] not in ('results',):
            task = parts[0]
    # Flatten metrics dict for DataFrame
    row = {
        'file': file,
        'method': method,
        'judge_model': judge_model,
        'task': task,
        'prompt_version': prompt_version,
        'aggregation': agg_type,
    }
    for k, v in metrics.items():
        row[k] = v
    rows.append(row)

# Create DataFrame
if rows:
    df = pd.DataFrame(rows)

    # Reorder columns
    cols = ['method', 'judge_model', 'task', 'prompt_version', 'aggregation', 'auc', 'file']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    # Print table
    # print(df.to_string(index=False))
    # Save as CSV
    out_csv = os.path.join(RESULTS_DIR, 'all_results_table.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved table to {out_csv}")
else:
    print("No metrics files found.")
