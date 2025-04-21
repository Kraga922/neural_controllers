import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle
import torch
from tqdm import tqdm
import gc

from datasets import load_dataset
import random
random.seed(0)

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']

def create_positive_negative_pairs(inputs, labels, max_pairs=None):
    """
    Creates pairs where each pair consists of one positive and one negative example.
    If one list is shorter, extends it by randomly sampling from itself.

    Args:
        inputs (list of str): The input texts.
        labels (list of int): Corresponding toxicity labels (1 for positive, 0 for negative).
        max_pairs (int): Maximum number of pairs to create.

    Returns:
        Tuple of (pairs, pair_labels)
    """
    if max_pairs is None:
        max_pairs = len(inputs)

    # Separate inputs by label
    positive_inputs = [inp for inp, lbl in zip(inputs, labels) if lbl == 1]
    negative_inputs = [inp for inp, lbl in zip(inputs, labels) if lbl == 0]

    # Debugging statements
    print(f"Number of positive examples: {len(positive_inputs)}")
    print(f"Number of negative examples: {len(negative_inputs)}")

    # Determine which list is longer
    max_length = max(len(positive_inputs), len(negative_inputs))
    num_pairs = min(max_length, max_pairs)

    # Extend shorter list if necessary
    if len(positive_inputs) < max_length:
        additional_needed = max_length - len(positive_inputs)
        positive_inputs.extend(random.choices(positive_inputs, k=additional_needed))
    elif len(negative_inputs) < max_length:
        additional_needed = max_length - len(negative_inputs)
        negative_inputs.extend(random.choices(negative_inputs, k=additional_needed))

    # Take only up to max_pairs
    paired_positive = positive_inputs[:num_pairs]
    paired_negative = negative_inputs[:num_pairs]

    # Create pairs as tuples (positive, negative)
    pairs = list(zip(paired_positive, paired_negative))

    # Create corresponding label pairs (1, 0)
    pair_labels = [[1, 0] for _ in range(len(pairs))]

    return pairs, pair_labels


def get_data(controller):
    # Load the dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat1123")

    train_inputs = [x['user_input'] for x in ds['train']]
    test_inputs = [x['user_input'] for x in ds['test']]
        
    # Format prompts using the controller
    train_inputs = [controller.format_prompt(x) for x in train_inputs]
    test_inputs = [controller.format_prompt(x) for x in test_inputs]
    
    # Extract labels
    train_labels = [x['toxicity'] for x in ds['train']]
    test_labels = [x['toxicity'] for x in ds['test']]
    
    return train_inputs, train_labels, test_inputs, test_labels


def split_states_on_idx(inputs, indices):
    if isinstance(inputs, list):
        return [inputs[i] for i in indices]
    elif isinstance(inputs, dict):
        split_states = {}
        for layer_idx, layer_states in inputs.items():
            split_states[layer_idx] = layer_states[indices]
        return split_states
    else:
        return inputs[indices]


def get_splits(k_folds, n_total, n_seeds):
    """
    Creates k-fold cross validation splits with multiple random seeds.
    
    Args:
        k_folds (int): Number of folds for cross validation
        n_total (int): Total number of samples
        n_seeds (int): Number of different random seeds
        
    Returns:
        List of dictionaries containing fold indices for each seed
    """
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/toxic_chat_results'
    os.makedirs(results_dir, exist_ok=True)
    out_name = f'{results_dir}/kfold_splits_k_{k_folds}_ntotal_{n_total}_nseeds_{n_seeds}.pkl'
    
    try:
        with open(out_name, 'rb') as f:
            splits = pickle.load(f)
            return splits
    except:
        pass

    splits = []
    
    # For each seed
    for seed in range(n_seeds):
        np.random.seed(seed)
        
        # Create random permutation of indices
        indices = np.random.permutation(n_total)
        
        # Calculate fold size
        fold_size = n_total // k_folds
        
        seed_splits = []
        # Create k folds
        for fold in range(k_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else n_total
            
            val_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            seed_splits.append({
                'val_indices': val_indices,
                'train_indices': train_indices,
                'fold': fold
            })
            
        splits.append(seed_splits)

    with open(out_name, 'wb') as f:
        pickle.dump(splits, f)

    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_components', type=int, default=1)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--rfm_iters', type=int, default=8)
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
    n_seeds = args.n_seeds
    n_components = args.n_components
    k_folds = args.k_folds
    
    if control_method not in ['rfm']:
        n_components=1
        
    use_logistic=(control_method=='logistic')
    unsupervised=(control_method in ['pca'])

    original_control_method = str(control_method)
    if control_method=='logistic_rfm':
        control_method='logistic'
        use_rfm=True
    elif control_method=='linear_rfm':
        control_method='linear'
        use_rfm=True
    elif control_method=='rfm_linear':
        control_method='rfm'
        use_rfm=False
    elif control_method=='rfm':
        use_rfm=True
    else:
        use_rfm=False
    
    print("Num components:", n_components)
                        
    language_model, tokenizer = load_model(model=model_name)
    controller = NeuralController(
        language_model,
        tokenizer,
        control_method=control_method,
        rfm_iters=args.rfm_iters,
        batch_size=1
    )  
    controller.name = model_name
    
    # Get base data first
    train_inputs, train_labels, test_inputs, test_labels = get_data(controller) 
    
    # Precompute and cache hidden states
    train_hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                   f'toxic_chat_train_{model_name}_unsupervised_{unsupervised}.pth')
    test_hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                  f'toxic_chat_test_{model_name}_unsupervised_{unsupervised}.pth')
    
    if os.path.exists(train_hidden_states_path) and os.path.exists(test_hidden_states_path):
        print("Loading cached hidden states...")
        with open(train_hidden_states_path, 'rb') as f:
            train_hidden_states = pickle.load(f)
        with open(test_hidden_states_path, 'rb') as f:
            test_hidden_states = pickle.load(f)
    else:
        print("Computing hidden states...")
        from direction_utils import get_hidden_states
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(train_hidden_states_path), exist_ok=True)
        
        # Compute train hidden states
        train_hidden_states = get_hidden_states(train_inputs, language_model, tokenizer, 
                                         controller.hidden_layers, 
                                         controller.hyperparams['forward_batch_size'])
        with open(train_hidden_states_path, 'wb') as f:
            pickle.dump(train_hidden_states, f)
            
        # Compute test hidden states
        test_hidden_states = get_hidden_states(test_inputs, language_model, tokenizer, 
                                        controller.hidden_layers, 
                                        controller.hyperparams['forward_batch_size'])
        with open(test_hidden_states_path, 'wb') as f:
            pickle.dump(test_hidden_states, f)
    
    # Convert train_labels to tensor for proper indexing
    train_labels = torch.tensor(train_labels).cuda().float()
    test_labels = torch.tensor(test_labels).cuda().float()
    
    # Get k-fold splits
    splits = get_splits(k_folds, len(train_labels), n_seeds)
    
    seed = args.seed
    seed_splits = splits[seed]
    
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/toxic_chat_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Store predictions from each fold
    all_test_predictions = []
    
    # For each fold
    for fold_data in seed_splits:
        val_indices = fold_data['val_indices']
        train_indices = fold_data['train_indices']
        fold = fold_data['fold']
        
        # Check if predictions for this fold already exist
        fold_predictions_file = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_fold_{fold}_test_predictions.pkl'
        if os.path.exists(fold_predictions_file):
            print(f"Loading cached predictions for fold {fold}")
            with open(fold_predictions_file, 'rb') as f:
                test_predictions = pickle.load(f)
            all_test_predictions.append(test_predictions)
            continue
            
        # Split the hidden states
        val_hidden_states = split_states_on_idx(train_hidden_states, val_indices)
        train_hidden_states_split = split_states_on_idx(train_hidden_states, train_indices)
        
        # Split the labels
        val_labels = train_labels[val_indices]
        train_labels_split = train_labels[train_indices]
        
        # If unsupervised, create pairs from the split data
        if unsupervised:
            # For unsupervised methods - implementation would need to change
            # to handle dictionary of hidden states rather than just inputs
            raise NotImplementedError("Unsupervised method not implemented for precomputed hidden states")
            
        try:
            controller.load(concept=f'toxic_chat_full_seed_{seed}_fold_{fold}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
        except:
            controller.compute_directions(train_hidden_states_split, train_labels_split)
            controller.save(concept=f'toxic_chat_full_seed_{seed}_fold_{fold}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')

        # Evaluate on validation and test sets
        val_metrics, test_metrics, test_predictions = controller.evaluate_directions(
            val_hidden_states, val_labels,
            test_hidden_states, test_labels,
            n_components=n_components,
            use_logistic=use_logistic,
            use_rfm=use_rfm,
            unsupervised=unsupervised,
            return_predictions=True
        )
        
        # Save fold-specific metrics and predictions
        out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_fold_{fold}_val_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(val_metrics, f)
            
        out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_fold_{fold}_test_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(test_metrics, f)
            
        # Save test predictions for this fold
        with open(fold_predictions_file, 'wb') as f:
            pickle.dump(test_predictions, f)
            
        # Store test predictions for this fold
        all_test_predictions.append(test_predictions)
    
    avg_test_predictions = np.mean(all_test_predictions, axis=0)

    avg_test_predictions_file = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_avg_test_predictions.pkl'
    with open(avg_test_predictions_file, 'wb') as f:
        pickle.dump(avg_test_predictions, f)
    
    # Compute and save AUC score
    auc_score = roc_auc_score(np.array(test_labels), avg_test_predictions.cpu().numpy())
    print(f"\nFinal AUC score across all folds: {auc_score:.4f}")
    
        
if __name__ == '__main__':              
    main()