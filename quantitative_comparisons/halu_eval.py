import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model

import numpy as np
import pickle
from tqdm import tqdm
import random
import json

random.seed(0)

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']

def read_hallucination_prompts_from_lines(lines):
    dicts = []
    for line in lines:
        x = json.loads(line)
        dicts.append(x)
    return dicts

def get_halu_eval_data(hal_type, prompt_version='v1'):
    if prompt_version == 'v1':
        qa_template = "Is the ANSWER to the following QUESTION factual? State yes or no.\n\n"
        qa_template += 'QUESTION: {question}\n\nANSWER: {answer}'

        general_template = "Is the RESPONSE to the following QUERY factual? State yes or no.\n\n"
        general_template += 'QUERY: {query}\n\nRESPONSE: {response}'

    if hal_type=='qa':
        data_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval/qa_data.txt'
        with open(data_path, 'r') as f:
            lines = f.readlines()
            raw_prompts = read_hallucination_prompts_from_lines(lines)

        n = len(raw_prompts)
        clean_train_prompts = raw_prompts[:int(n//2)]
        clean_eval_prompts = raw_prompts[int(n//2):]
        
        # Generate training data
        train_inputs = []
        train_labels = []
        for prompt in clean_train_prompts:
            x_true = qa_template.format(question=prompt['question'], answer=prompt['right_answer'])
            x_false = qa_template.format(question=prompt['question'], answer=prompt['hallucinated_answer'])
            train_inputs.append(x_true)
            train_inputs.append(x_false)
            train_labels += [0,1]
        
        test_inputs = []
        test_labels = []
        for prompt in clean_eval_prompts:
            x_true = wrapped_qa_template.format(knowledge=prompt['knowledge'], question=prompt['question'], answer=prompt['right_answer'])
            x_false = wrapped_qa_template.format(knowledge=prompt['knowledge'], question=prompt['question'], answer=prompt['hallucinated_answer'])
            test_inputs.append(x_true)
            test_inputs.append(x_false)
            test_labels += [0,1]
            
    elif hal_type=='general':
        # Get QA data for training
        qa_data_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval/qa_data.txt'
        with open(qa_data_path, 'r') as f:
            lines = f.readlines()
            raw_train_prompts = read_hallucination_prompts_from_lines(lines)
            
        n = len(raw_train_prompts)
        train_prompts = raw_train_prompts[:int(n//2)]
        
        # Generate training data from QA data
        train_inputs = []
        train_labels = []
        for prompt in train_prompts:
            x_true = qa_template.format(knowledge=prompt['knowledge'], question=prompt['question'], answer=prompt['right_answer'])
            x_false = qa_template.format(knowledge=prompt['knowledge'], question=prompt['question'], answer=prompt['hallucinated_answer'])
            train_inputs.append(x_true)
            train_inputs.append(x_false)
            train_labels += [0,1]
            
        # Get general data for evaluation
        data_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval/general_data.txt'
        with open(data_path, 'r') as f:
            lines = f.readlines()
            eval_prompts = read_hallucination_prompts_from_lines(lines)
            
        test_inputs = []
        test_labels = []
        for prompt in eval_prompts:
            x = general_template.format(query=prompt['user_query'], response=prompt['chatgpt_response'])
            test_inputs.append(x)
            test_labels.append(int(prompt['hallucination'].lower().strip() == 'yes'))

    return train_inputs, np.array(train_labels), test_inputs, np.array(test_labels)

def get_splits(n_val, n_total, n_seeds, hal_type):
    """
    Generate validation splits for the test set instead of training set.
    n_val: number of validation samples
    n_total: total number of test samples
    n_seeds: number of different random seeds for splits
    hal_type: type of hallucination data
    """
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halu_eval_results'
    os.makedirs(results_dir, exist_ok=True)
    out_name = f'{results_dir}/{hal_type}_test_splits_nval_{n_val}_ntotal_{n_total}_nseeds_{n_seeds}.pkl'
    try:
        with open(out_name, 'rb') as f:
            splits = pickle.load(f)
            return splits
    except:
        pass

    splits = []
    indices = np.arange(n_total)

    for seed in range(n_seeds):
        np.random.seed(seed)
        val_indices = np.random.choice(indices, size=n_val, replace=False)
        test_indices = np.setdiff1d(indices, val_indices)

        splits.append({
            'test_indices': test_indices,
            'val_indices': val_indices
        })

    with open(out_name, 'wb') as f:
        pickle.dump(splits, f)

    return splits

def split_test_states_on_idx(inputs, split):
    val_inputs, test_inputs = {}, {}
    for layer_idx, layer_states in inputs.items():
        val_inputs[layer_idx] = layer_states[split['val_indices']]
        test_inputs[layer_idx] = layer_states[split['test_indices']]
    return val_inputs, test_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it')
    parser.add_argument('--n_components', type=int, default=1)
    parser.add_argument('--hal_type', type=str, default='qa')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--prompt_version', type=str, default='v1')
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
    n_components = args.n_components
    hal_type = args.hal_type
    prompt_version = args.prompt_version

    if control_method not in ['rfm']:
        n_components=1
    
    print("Num components:", n_components)
    
    language_model, tokenizer = load_model(model=model_name)
    
    controller = NeuralController(
            language_model,
            tokenizer,
            control_method=control_method,
            rfm_iters=5,
            batch_size=1
    )
    unformatted_train_inputs, train_labels, unformatted_test_inputs, test_labels = get_halu_eval_data(hal_type, prompt_version)

    train_inputs = [controller.format_prompt(x) for x in unformatted_train_inputs]
    test_inputs = [controller.format_prompt(x) for x in unformatted_test_inputs]

    print("="*100)
    print(train_inputs[0])
    print("="*100)
    print(train_labels[0])
    print("="*100)
    print(test_inputs[0])
    print("="*100)
    print(test_labels[0])
    print("="*100)
    train_hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                      f'halu_eval_{hal_type}_{model_name}_train.pth')
    if os.path.exists(train_hidden_states_path):
        with open(train_hidden_states_path, 'rb') as f:
            train_hidden_states = pickle.load(f)
    else:
        from direction_utils import get_hidden_states
        train_hidden_states = get_hidden_states(train_inputs, language_model, tokenizer, 
                                      controller.hidden_layers, 
                                      controller.hyperparams['forward_batch_size'])
    
        with open(train_hidden_states_path, 'wb') as f:
            pickle.dump(train_hidden_states, f)

    print("Getting test hidden states")
    test_hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                      f'halu_eval_{hal_type}_{model_name}_test.pth')
    if os.path.exists(test_hidden_states_path):
        with open(test_hidden_states_path, 'rb') as f:
            test_hidden_states = pickle.load(f) 
    else:
        from direction_utils import get_hidden_states
        test_hidden_states = get_hidden_states(test_inputs, language_model, tokenizer, 
                                      controller.hidden_layers, 
                                      controller.hyperparams['forward_batch_size'])
    
        with open(test_hidden_states_path, 'wb') as f:
            pickle.dump(test_hidden_states, f)

    if hal_type in ['general', 'qa']:
        concept = f'halu_eval_detect'
    else:
        concept = f'halu_eval_detect_{hal_type}'

    try:
        print("Loading directions")
        controller.load(concept=concept, model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
    except:
        print("Computing directions")
        controller.compute_directions(train_hidden_states, train_labels)
        controller.save(concept=concept, model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')

    # Changed to split the test set instead of train set
    splits = get_splits(n_val=len(test_inputs)//2, n_total=len(test_inputs), n_seeds=n_seeds, hal_type=hal_type)
    
    for seed in tqdm(range(n_seeds)):
        split = splits[seed]
        
        # Apply split to test data instead of train data
        print("Splitting test data")
        val_hidden_states_on_seed, test_hidden_states_on_seed = split_test_states_on_idx(test_hidden_states, split)
        val_labels_on_seed, test_labels_on_seed = test_labels[split['val_indices']], test_labels[split['test_indices']]

        print("Evaluating directions")
        val_metrics, test_metrics, _, _ = controller.evaluate_directions(
            val_hidden_states_on_seed, val_labels_on_seed,
            test_hidden_states_on_seed, test_labels_on_seed,
            n_components=n_components,
        )
        print("Done evaluating directions")

        print("Saving results")
        results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halu_eval_results'
        os.makedirs(results_dir, exist_ok=True)
        out_name = f'{results_dir}/{model_name}_{control_method}_seed_{seed}_{hal_type}_prompt_{prompt_version}_val_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(val_metrics, f)

        out_name = f'{results_dir}/{model_name}_{control_method}_seed_{seed}_{hal_type}_prompt_{prompt_version}_test_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(test_metrics, f)
        print("Done saving results")
        
if __name__ == '__main__':              
    main()