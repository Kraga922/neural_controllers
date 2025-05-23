import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
import json

def count_halubench():
    print("\nHaluBench Dataset:")
    ds = load_dataset("PatronusAI/HaluBench")['test']
    
    # Count total samples
    total = len(ds)
    print(f"Total samples: {total}")
    
    # Count by source dataset and labels
    sources = {}
    labels = {'PASS': 0, 'FAIL': 0}
    source_labels = {}
    
    for item in ds:
        source = item['source_ds']
        label = item['label']
        
        # Update overall counts
        sources[source] = sources.get(source, 0) + 1
        labels[label] = labels.get(label, 0) + 1
        
        # Update source-specific label counts
        if source not in source_labels:
            source_labels[source] = {'PASS': 0, 'FAIL': 0}
        source_labels[source][label] += 1
    
    print("\nSamples by source with label distribution:")
    for source in sorted(sources.keys()):
        total_source = sources[source]
        pass_count = source_labels[source]['PASS']
        fail_count = source_labels[source]['FAIL']
        pass_pct = (pass_count / total_source) * 100
        fail_pct = (fail_count / total_source) * 100
        
        print(f"\n{source}:")
        print(f"Total: {total_source}")
        print(f"- Non-hallucinated (PASS): {pass_count} ({pass_pct:.1f}%)")
        print(f"- Hallucinated (FAIL): {fail_count} ({fail_pct:.1f}%)")
        
    print("\nOverall label distribution:")
    total_samples = sum(labels.values())
    pass_pct = (labels['PASS'] / total_samples) * 100
    fail_pct = (labels['FAIL'] / total_samples) * 100
    print(f"Non-hallucinated (PASS): {labels['PASS']} ({pass_pct:.1f}%)")
    print(f"Hallucinated (FAIL): {labels['FAIL']} ({fail_pct:.1f}%)")

def count_fava():
    print("\nFAVA Dataset:")
    
    # Load annotated data
    try:
        neural_controllers_dir = os.environ.get('NEURAL_CONTROLLERS_DIR', '')
        if neural_controllers_dir:
            file_path = f'{neural_controllers_dir}/data/hallucinations/fava/annotations.json'
            with open(file_path, 'r') as file:
                annotated_data = json.load(file)
            
            # Count labels in annotated data
            hal_count = 0
            for item in annotated_data:
                s = item['annotated']
                # Check if any hallucination tags exist
                if any(tag in s for tag in ["<entity>", "<relation>", "<sentence>", "<invented>", "<subjective>", "<unverifiable>"]):
                    hal_count += 1
                    
            print(f"Annotated samples: {len(annotated_data)}")
            print(f"- With hallucinations: {hal_count}")
            print(f"- Without hallucinations: {len(annotated_data) - hal_count}")
    except Exception as e:
        print(f"Could not load annotated data: {e}")
    
    # Load training data
    try:
        ds = load_dataset("fava-uw/fava-data")
        print(f"Training samples: {len(ds['train'])}")
    except Exception as e:
        print(f"Could not load training data: {e}")

def count_toxic_chat():
    print("\nToxic Chat Dataset:")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat1123")
    
    # Count total samples
    train_total = len(ds['train'])
    test_total = len(ds['test'])
    print(f"Total training samples: {train_total}")
    print(f"Total test samples: {test_total}")
    
    # Count labels in training set
    train_toxic = sum(1 for x in ds['train'] if x['toxicity'])
    train_non_toxic = train_total - train_toxic
    train_toxic_pct = (train_toxic / train_total) * 100
    train_non_toxic_pct = (train_non_toxic / train_total) * 100
    
    print("\nTraining set label distribution:")
    print(f"- Toxic: {train_toxic} ({train_toxic_pct:.1f}%)")
    print(f"- Non-toxic: {train_non_toxic} ({train_non_toxic_pct:.1f}%)")
    
    # Count labels in test set
    test_toxic = sum(1 for x in ds['test'] if x['toxicity'])
    test_non_toxic = test_total - test_toxic
    test_toxic_pct = (test_toxic / test_total) * 100
    test_non_toxic_pct = (test_non_toxic / test_total) * 100
    
    print("\nTest set label distribution:")
    print(f"- Toxic: {test_toxic} ({test_toxic_pct:.1f}%)")
    print(f"- Non-toxic: {test_non_toxic} ({test_non_toxic_pct:.1f}%)")

def count_fava_training():
    print("\nFAVA Training Dataset:")
    ds = load_dataset("fava-uw/fava-data")
    
    # Limit to first 10k samples
    train_total = min(10000, len(ds['train']))
    print(f"Processing first {train_total} training samples (out of {len(ds['train'])} total)")
    
    # Count hallucinations in training data
    from bs4 import BeautifulSoup
    _TAGS = ["entity", "relation", "sentence", "invented", "subjective", "unverifiable"]
    
    tag_counts = {tag: 0 for tag in _TAGS}
    total_hallucinated = 0
    
    for i, item in enumerate(ds['train']):
        if i >= train_total:  # Stop after 10k samples
            break
            
        completion = item['completion']
        soup = BeautifulSoup(completion, "html.parser")
        
        has_hallucination = False
        for tag in _TAGS:
            count = len(soup.find_all(tag))
            tag_counts[tag] += count
            if count > 0:
                has_hallucination = True
        
        if has_hallucination:
            total_hallucinated += 1
    
    print(f"\nSamples with hallucinations: {total_hallucinated} ({(total_hallucinated/train_total)*100:.1f}%)")
    print(f"Samples without hallucinations: {train_total - total_hallucinated} ({((train_total-total_hallucinated)/train_total)*100:.1f}%)")
    
    print("\nHallucination type distribution:")
    for tag, count in tag_counts.items():
        print(f"- {tag}: {count} instances")

def count_halu_eval_wild():
    print("\nHaluEval Wild Dataset:")
    try:
        neural_controllers_dir = os.environ.get('NEURAL_CONTROLLERS_DIR', '')
        if neural_controllers_dir:
            data_path = f'{neural_controllers_dir}/data/hallucinations/halu_eval_wild/HaluEval_Wild_6types.json'
            with open(data_path, 'r') as file:
                data = json.load(file)
            
            # Count total samples
            total = len(data)
            print(f"Total samples: {total}")
            
            # Count by type
            types = {}
            for item in data:
                qtype = item['query_type']
                types[qtype] = types.get(qtype, 0) + 1
            
            print("\nSamples by type:")
            for qtype, count in types.items():
                print(f"{qtype}: {count}")
            
            # Note: HaluEval Wild is a classification dataset where each example belongs to one of six categories
            # There's no binary hallucination/non-hallucination split
    except Exception as e:
        print(f"Could not load HaluEval Wild data: {e}")

def count_halu_eval():
    print("\nHaluEval Dataset:")
    try:
        neural_controllers_dir = os.environ.get('NEURAL_CONTROLLERS_DIR', '')
        if neural_controllers_dir:
            # Count QA data
            qa_path = f'{neural_controllers_dir}/data/hallucinations/halu_eval/qa_data.txt'
            with open(qa_path, 'r') as f:
                qa_lines = f.readlines()
                qa_data = [json.loads(line) for line in qa_lines]
            print(f"QA samples: {len(qa_data)} original questions")
            print(f"- Non-hallucinated answers: {len(qa_data)} (correct answers)")
            print(f"- Hallucinated answers: {len(qa_data)} (incorrect answers)")
            print(f"Total QA pairs: {len(qa_data) * 2}")
            
            # Count general data
            general_path = f'{neural_controllers_dir}/data/hallucinations/halu_eval/general_data.txt'
            with open(general_path, 'r') as f:
                general_lines = f.readlines()
                general_data = [json.loads(line) for line in general_lines]
            
            # Count hallucinations in general data
            hal_count = sum(1 for item in general_data if item['hallucination'].lower().strip() == 'yes')
            print(f"\nGeneral samples: {len(general_data)}")
            print(f"- With hallucinations: {hal_count}")
            print(f"- Without hallucinations: {len(general_data) - hal_count}")
            
    except Exception as e:
        print(f"Could not load HaluEval data: {e}")

if __name__ == "__main__":
    print("Getting dataset statistics...")
    
    try:
        count_halubench()
    except Exception as e:
        print(f"Error loading HaluBench: {e}")
        
    try:
        count_fava()
    except Exception as e:
        print(f"Error loading FAVA: {e}")
        
    try:
        count_toxic_chat()
    except Exception as e:
        print(f"Error loading Toxic Chat: {e}")
        
    try:
        count_fava_training()
    except Exception as e:
        print(f"Error loading FAVA training: {e}")
        
    try:
        count_halu_eval_wild()
    except Exception as e:
        print(f"Error loading HaluEval Wild: {e}")
        
    try:
        count_halu_eval()
    except Exception as e:
        print(f"Error loading HaluEval: {e}")
