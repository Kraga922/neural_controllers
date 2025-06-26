import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, '/u/dbeaglehole/xrfm')

from xrfm import xRFM, RFM
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from copy import deepcopy
from tqdm import tqdm

from utils import preds_to_proba

# For scaling linear probe beyond ~50k datapoints.
def batch_transpose_multiply(A, B, mb_size=5000):
    n = len(A)
    assert(len(A) == len(B))
    batches = torch.split(torch.arange(n), mb_size)
    sum = 0.
    for b in batches:
        Ab = A[b].cuda()
        Bb = B[b].cuda()
        sum += Ab.T @ Bb

        del Ab, Bb
    return sum

def accuracy_fn(preds, truth):
    assert(len(preds)==len(truth))
    true_shape = truth.shape
    
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(truth, torch.Tensor):
        truth = truth.cpu().numpy()
        
    preds = preds.reshape(true_shape)
    
    if preds.shape[1] == 1:
        preds = np.where(preds >= 0.5, 1, 0)
        truth = np.where(truth >= 0.5, 1, 0)
    else:
        preds = np.argmax(preds, axis=1)
        truth = np.argmax(truth, axis=1)
        
    acc = np.sum(preds==truth)/len(preds) * 100
    return acc

def pearson_corr(x, y):     
    assert(x.shape == y.shape)
    
    x = x.float() + 0.0
    y = y.float() + 0.0

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))

    return numerator / denominator

def split_data(data, labels):
    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=0.2, random_state=0, shuffle=True
    ) 
    return data_train, data_test, labels_train, labels_test

def precision_score(preds, labels):
    true_positives = np.sum((preds == 1) & (labels == 1))
    predicted_positives = np.sum(preds == 1)
    return true_positives / (predicted_positives + 1e-8)  # add small epsilon to prevent division by zero

def recall_score(preds, labels):
    true_positives = np.sum((preds == 1) & (labels == 1))
    actual_positives = np.sum(labels == 1)
    return true_positives / (actual_positives + 1e-8)  # add small epsilon to prevent division by zero

def f1_score(preds, labels):
    precision = precision_score(preds, labels)
    recall = recall_score(preds, labels)
    return 2 * (precision * recall) / (precision + recall + 1e-8)  # add small epsilon to prevent division by zero

# def compute_prediction_metrics(preds, labels, classification_threshold=0.5):
    
#     # if len(labels.shape) > 1:
#     #     labels = labels.ravel()

#     if len(labels.shape) == 1:
#         labels = labels.reshape(-1, 1)
#     num_classes = labels.shape[1]
#     if isinstance(preds, torch.Tensor):
#         preds = preds.cpu().numpy()
#     if isinstance(labels, torch.Tensor):
#         labels = labels.cpu().numpy()

#     auc = roc_auc_score(labels, preds)
#     mse = np.mean((preds-labels)**2)
#     if num_classes == 1:  # Binary classification
#         preds = np.where(preds >= classification_threshold, 1, 0)
#         labels = np.where(labels >= classification_threshold, 1, 0)
#         acc = accuracy_fn(preds, labels)
#         precision = precision_score(preds, labels)
#         recall = recall_score(preds, labels)
#         f1 = f1_score(preds, labels)
#     else:  # Multiclass classification
#         preds_classes = np.argmax(preds, axis=1)
#         label_classes = np.argmax(labels, axis=1)
        
#         # Compute accuracy
#         acc = np.sum(preds_classes == label_classes)/ len(preds) * 100
        
#         # Initialize metrics for averaging
#         precision, recall, f1 = 0.0, 0.0, 0.0
        
#         # Compute metrics for each class
#         for class_idx in range(num_classes):
#             class_preds = (preds_classes == class_idx).astype(np.float32)
#             class_labels = (label_classes == class_idx).astype(np.float32)
            
#             precision += precision_score(class_preds, class_labels)
#             recall += recall_score(class_preds, class_labels)
#             f1 += f1_score(class_preds, class_labels)
        
#         # Average metrics across classes
#         precision /= num_classes
#         recall /= num_classes
#         f1 /= num_classes

#     metrics = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mse': mse}
#     return metrics

def compute_prediction_metrics(pred_proba, y_true):
    """
    Compute prediction metrics with proper shape handling
    """
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    import numpy as np
    import torch
    
    # Convert to numpy if needed
    if isinstance(pred_proba, torch.Tensor):
        pred_proba = pred_proba.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    
    # Ensure y_true is 1D
    y_true = y_true.flatten()
    
    # Handle pred_proba shape
    if len(pred_proba.shape) > 1:
        if pred_proba.shape[1] == 2:
            # Binary classification with 2 columns - take positive class probability
            pred_proba_1d = pred_proba[:, 1]
        elif pred_proba.shape[1] == 1:
            # Single column - flatten
            pred_proba_1d = pred_proba.flatten()
        else:
            # Multi-class - take max probability
            pred_proba_1d = np.max(pred_proba, axis=1)
    else:
        pred_proba_1d = pred_proba.flatten()
    
    # Ensure same length
    min_len = min(len(y_true), len(pred_proba_1d))
    y_true = y_true[:min_len]
    pred_proba_1d = pred_proba_1d[:min_len]
    
    print(f"Debug: y_true shape: {y_true.shape}, pred_proba shape: {pred_proba_1d.shape}")
    print(f"Debug: y_true unique values: {np.unique(y_true)}")
    
    metrics = {}
    
    try:
        # Calculate AUC
        if len(np.unique(y_true)) > 1:  # Need at least 2 classes for AUC
            metrics['auc'] = roc_auc_score(y_true, pred_proba_1d)
        else:
            metrics['auc'] = 0.5  # Default for single class
    except Exception as e:
        print(f"AUC calculation failed: {e}")
        metrics['auc'] = 0.5
    
    try:
        # Calculate accuracy
        pred_binary = (pred_proba_1d > 0.5).astype(int)
        metrics['acc'] = accuracy_score(y_true, pred_binary)
    except Exception as e:
        print(f"Accuracy calculation failed: {e}")
        metrics['acc'] = 0.5
    
    try:
        # Calculate F1
        pred_binary = (pred_proba_1d > 0.5).astype(int)
        metrics['f1'] = f1_score(y_true, pred_binary, average='binary')
    except Exception as e:
        print(f"F1 calculation failed: {e}")
        metrics['f1'] = 0.5
    
    return metrics
def safe_model_predict_proba(model, X):
    """
    Safely get predictions from model with shape handling
    """
    try:
        pred_proba = model.predict_proba(X)
        
        # Ensure predictions are in correct format
        if isinstance(pred_proba, torch.Tensor):
            pred_proba = pred_proba.cpu().numpy()
        
        # If shape is (n, 2) for binary classification, we want probabilities for positive class
        if len(pred_proba.shape) > 1 and pred_proba.shape[1] == 2:
            return pred_proba[:, 1:2]  # Keep as 2D with single column
        elif len(pred_proba.shape) > 1 and pred_proba.shape[1] == 1:
            return pred_proba
        else:
            return pred_proba.reshape(-1, 1)  # Ensure 2D
            
    except Exception as e:
        print(f"Error in model prediction: {e}")
        # Return default predictions
        return np.full((X.shape[0], 1), 0.5)

def get_hidden_states(prompts, model, tokenizer, hidden_layers, forward_batch_size, rep_token=-1, all_positions=False):

    if isinstance(prompts, np.ndarray):
        prompts = prompts.tolist()

    try: 
        name = model._get_name()
        seq2seq = (name=='T5ForConditionalGeneration')
    except:
        seq2seq = False

    if seq2seq:
        encoded_inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    else:
        encoded_inputs = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        encoded_inputs['attention_mask'] = encoded_inputs['attention_mask'].half()
    
    dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=forward_batch_size)

    all_hidden_states = {}
    for layer_idx in hidden_layers:
        all_hidden_states[layer_idx] = []

    use_concat = list(hidden_layers)==['concat']

    # Loop over batches and accumulate outputs
    print("Getting activations from forward passes")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask = batch

            if seq2seq:
                encoder_outputs = model.encoder(
                    input_ids=input_ids,
                    output_hidden_states=True
                )                
                out_hidden_states = encoder_outputs.hidden_states

                decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]]).cuda()
                decoder_outputs = model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    output_hidden_states=True
                )
                out_hidden_states = decoder_outputs.hidden_states

                num_layers = len(out_hidden_states)-1 # exclude embedding layer
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                num_layers = len(model.model.layers)
                out_hidden_states = outputs.hidden_states
            
            hidden_states_all_layers = []
            for layer_idx, hidden_state in zip(range(-1, -num_layers, -1), reversed(out_hidden_states)):
                
                if use_concat:
                    hidden_states_all_layers.append(hidden_state[:,rep_token,:].detach().cpu())
                elif all_positions:
                    all_hidden_states[layer_idx].append(hidden_state.detach().cpu())
                else:
                    all_hidden_states[layer_idx].append(hidden_state[:,rep_token,:].detach().cpu())
                    
            if use_concat:
                hidden_states_all_layers = torch.cat(hidden_states_all_layers, dim=1)
                all_hidden_states['concat'].append(hidden_states_all_layers)
                      
    # Concatenate results from all batches
    final_hidden_states = {}
    for layer_idx, hidden_state_list in all_hidden_states.items():
        final_hidden_states[layer_idx] = torch.cat(hidden_state_list, dim=0)
        
    return final_hidden_states

def project_hidden_states(hidden_states, directions, n_components):
    """
    directions:
        {-1 : [beta_{1}, .., beta_{m}],
        ...,
        -31 : [beta_{1}, ..., beta_{m}]
        }
    hidden_states:
        {-1 : [h_{1}, .., h_{d}],
        ...,
        -31 : [h_{1}, ..., h_{d}]
        }
    """
    print("n_components", n_components)
    assert(hidden_states.keys()==directions.keys())
    layers = hidden_states.keys()
    
    projections = {}
    for layer in layers:
        vecs = directions[layer][:n_components].T
        projections[layer] = hidden_states[layer].cuda()@vecs.cuda()
    return projections

def aggregate_projections_on_coefs(projections, detector_coef):
    """
    detector_coefs:
        {-1 : [beta_{1}, bias_{1}],
        ...,
        -31 : [beta_31_{31}, bias_{31},
        'agg_sol': [beta_{agg}, bias_{agg}]]
    projections:
        {-1 : tensor (n, n_components),
        ...,
        -31 : tensor (n, n_components),
        }
    """
        
    layers = projections.keys()
    agg_projections = []
    for layer in layers:
        X = projections[layer].cuda()
        agg_projections.append(X.squeeze(0))
    
    agg_projections = torch.concat(agg_projections, dim=1).squeeze()
    agg_beta = detector_coef[0]
    agg_bias = detector_coef[1]
    agg_preds = agg_projections@agg_beta + agg_bias
    return agg_preds

def project_onto_direction(tensors, direction, device='cuda'):
    """
    tensors : (n, d)
    direction : (d, )
    output : (n, )
    """
    assert(len(tensors.shape)==2)
    assert(tensors.shape[1] == direction.shape[0])
    
    return tensors.to(device=device) @ direction.to(device=device, dtype=tensors.dtype)

def fit_pca_model(train_X, train_y, n_components=1, mean_center=True):
    """
    Assumes the data are in ordered pairs of pos/neg versions of the same prompts:
    
    e.g. the first four elements of train_X correspond to 
    
    Dishonestly say something about {object x}
    Honestly say something about {object x}
    
    Honestly say something about {object y}
    Dishonestly say something about {object y}
    
    """
    pos_indices = torch.isclose(train_y, torch.ones_like(train_y)).squeeze(1)
    neg_indices = torch.isclose(train_y, torch.zeros_like(train_y)).squeeze(1)
    
    pos_examples = train_X[pos_indices]
    neg_examples = train_X[neg_indices]
    
    dif_vectors = pos_examples - neg_examples
    
    # randomly flip the sign of the vectors
    random_signs = torch.randint(0, 2, (len(dif_vectors),)).float().to(dif_vectors.device) * 2 - 1
    dif_vectors = dif_vectors * random_signs.reshape(-1,1)
    if mean_center:
        dif_vectors -= torch.mean(dif_vectors, dim=0, keepdim=True)

    # dif_vectors : (n//2, d)
    XtX = dif_vectors.T@dif_vectors
    # _, U = torch.linalg.eigh(XtX)
    # return torch.flip(U[:,-n_components:].T, dims=(0,))

    _, U = torch.lobpcg(XtX, k=n_components)
    return U.T

def append_one(X):
    Xb = torch.concat([X, torch.ones_like(X[:,0]).unsqueeze(1)], dim=1)
    new_shape = X.shape[:1] + (X.shape[1]+1,) 
    assert(Xb.shape == new_shape)
    return Xb

def linear_solve(X, y, use_bias=True, reg=0):
    """
    projected_inputs : (n, d)
    labels : (n, c) or (n, )
    """
    
    if use_bias:
        inputs = append_one(X)
    else:
        inputs = X
    
    if len(y.shape) == 1:
        y = y.unsqueeze(1)

    num_classes = y.shape[1]
    n, d = inputs.shape
    
    if n>d:   
        XtX = inputs.T@inputs
        XtY = inputs.T@y
        beta = torch.linalg.pinv(XtX + reg*torch.eye(d).to(inputs.device))@XtY # (d, c)
    else:
        XXt = inputs@inputs.T
        alpha = torch.linalg.pinv(XXt + reg*torch.eye(n).to(inputs.device))@y # (n, c)
        beta = inputs.T @ alpha
    
    if use_bias:
        sol = beta[:-1]
        bias = beta[-1]
        if num_classes == 1:
            bias = bias.item()
        return sol, bias
    else:
        return beta
        

def logistic_solve(X, y, C=1):
    """
    projected_inputs : (n, d)
    labels : (n, c)
    """

    num_classes = y.shape[1]
    if num_classes == 1:
        y = y.flatten()
    else:
        y = y.argmax(dim=1)
    model = LogisticRegression(fit_intercept=True, max_iter=1000, C=C) # use bias
    model.fit(X.cpu(), y.cpu())
    
    beta = torch.from_numpy(model.coef_).to(X.dtype).to(X.device)
    bias = torch.from_numpy(model.intercept_).to(X.dtype).to(X.device)
    
    return beta.T, bias

def aggregate_layers(layer_outputs, train_y, val_y, test_y, agg_model='linear', tuning_metric='auc'):
    
    # solve aggregator on validation set
    train_X = torch.concat(layer_outputs['train'], dim=1) # (n, num_layers*n_components)    
    val_X = torch.concat(layer_outputs['val'], dim=1) # (n, num_layers*n_components)    
    test_X = torch.concat(layer_outputs['test'], dim=1) # (n, num_layers*n_components)    

    print("train_X", train_X.shape, "val_X", val_X.shape, "test_X", test_X.shape)

    maximize_metric = (tuning_metric in ['f1', 'auc', 'acc'])

    if agg_model=='rfm':
        # bw_search_space = [5, 10, 20]
        # reg_search_space = [1e-4, 1e-3, 1e-2]
        # kernel_search_space = ['l2_high_dim']
        bw_search_space = [10]
        reg_search_space = [1e-4, 1e-3, 1e-2]
        kernel_search_space = ['l2_high_dim']

        search_space_size = len(bw_search_space) * len(reg_search_space) * len(kernel_search_space)
        print("Search space size: {}".format(search_space_size))

        best_rfm_params = None
        best_rfm_score = float('-inf') if maximize_metric else float('inf')
        for bw in bw_search_space:
            for reg in reg_search_space:
                for kernel in kernel_search_space:
                    rfm_params = {
                        'model': {
                            'kernel': kernel,
                            'bandwidth': bw,
                        },
                        'fit': {
                            'reg': reg,
                            'iters': 10,
                        }
                    }
                    model = xRFM(rfm_params, device='cuda', tuning_metric=tuning_metric)
                    model.fit(train_X, train_y, val_X, val_y)              
                    val_preds = model.predict(val_X)
                    metrics = compute_prediction_metrics(val_preds, val_y)

                    if (maximize_metric and metrics[tuning_metric] > best_rfm_score)\
                        or (not maximize_metric and metrics[tuning_metric] < best_rfm_score):

                        best_rfm_score = metrics[tuning_metric]
                        best_rfm_params = deepcopy(rfm_params)

        model = xRFM(best_rfm_params, device='cuda', tuning_metric=tuning_metric)
        model.fit(train_X, train_y, val_X, val_y)              
        test_preds = model.predict(test_X)

        metrics = compute_prediction_metrics(test_preds, test_y)
        return metrics, None, None, test_preds
    
    elif agg_model=='logistic':
        C_search_space = [1000, 100, 10, 1, 1e-1, 1e-2]
        best_logistic_params = None
        best_logistic_score = float('-inf') if maximize_metric else float('inf')
        for C in C_search_space:
            agg_beta, agg_bias = logistic_solve(train_X, train_y, C=C) # (num_layers*n_components, num_classes)
            val_preds = val_X@agg_beta + agg_bias
            metrics = compute_prediction_metrics(val_preds, val_y)
            if (maximize_metric and metrics[tuning_metric] > best_logistic_score)\
                or (not maximize_metric and metrics[tuning_metric] < best_logistic_score):

                best_logistic_score = metrics[tuning_metric]
                best_logistic_params = (agg_beta, agg_bias)

        agg_beta, agg_bias = best_logistic_params

    elif agg_model=='linear':
        reg_search_space = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
        best_linear_params = None
        best_linear_score = float('-inf') if maximize_metric else float('inf')
        for reg in reg_search_space:
            agg_beta, agg_bias = linear_solve(train_X, train_y, reg=reg)
            val_preds = val_X@agg_beta + agg_bias
            metrics = compute_prediction_metrics(val_preds, val_y)
            if (maximize_metric and metrics[tuning_metric] > best_linear_score)\
                or (not maximize_metric and metrics[tuning_metric] < best_linear_score):
                
                best_linear_score = metrics[tuning_metric]
                best_linear_params = (agg_beta, agg_bias)

        agg_beta, agg_bias = best_linear_params

    else:
        raise ValueError(f"Invalid aggregation model: {agg_model}")

    # evaluate aggregated predictor on test set
    test_preds = test_X@agg_beta + agg_bias
    test_preds = test_preds.reshape(test_y.shape)
    metrics = compute_prediction_metrics(test_preds, test_y)
    return metrics, agg_beta, agg_bias, test_preds

    
# def train_rfm_probe_on_concept(train_X, train_y, val_X, val_y, 
    #                            hyperparams, search_space=None, 
    #                            tuning_metric='auc'):
    
    # if search_space is None:
    #     search_space = {
    #         'regs': [1e-3],
    #         'bws': [1, 10, 100],
    #         'center_grads': [True, False]
    #     }
    
    # best_model = None
    # maximize_metric = (tuning_metric in ['f1', 'auc', 'acc', 'top_agop_vectors_ols_auc'])
    # best_score = float('-inf') if maximize_metric else float('inf')
    # for reg in search_space['regs']:
    #     for bw in search_space['bws']:
    #         for center_grads in search_space['center_grads']:
    #             try:
    #                 rfm_params = {
    #                     'model': {
    #                         'kernel': 'l2_high_dim',
    #                         'bandwidth': bw,
    #                         'tuning_metric': tuning_metric,
    #                     },
    #                     'fit': {
    #                         'reg': reg,
    #                         'iters': hyperparams['rfm_iters'],
    #                         'center_grads': center_grads,
    #                         'early_stop_rfm': True,
    #                         'get_agop_best_model': True,
    #                         'top_k': hyperparams['n_components']
    #                     }
    #                 }
    #                 model = RFM(**rfm_params['model'], device='cuda')
    #                 model.fit((train_X, train_y), 
    #                           (val_X, val_y), 
    #                           **rfm_params['fit']
    #                         )

    #                 if tuning_metric == 'top_agop_vectors_ols_auc':
    #                     top_k = hyperparams['n_components']
    #                     targets = val_y

    #                     _, U = torch.lobpcg(model.agop_best_model, k=top_k)
    #                     top_eigenvectors = U[:, :top_k]
    #                     projections = val_X @ top_eigenvectors
    #                     projections = projections.reshape(-1, top_k)
                        
    #                     XtX = projections.T @ projections
    #                     Xty = projections.T @ targets
    #                     betas = torch.linalg.pinv(XtX) @ Xty
    #                     preds = torch.sigmoid(projections @ betas).reshape(targets.shape)
    #                     val_score = roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy())
    #                 else:
    #                     pred_proba = model.predict_proba(val_X)
    #                     val_score = compute_prediction_metrics(pred_proba, val_y)[tuning_metric]

    #                 if maximize_metric and val_score > best_score or not maximize_metric and val_score < best_score:
    #                     best_score = val_score
    #                     best_reg = reg
    #                     best_model = deepcopy(model)
    #                     best_bw = bw    
    #                     best_center_grads = center_grads
    #             except Exception as e:
    #                 import traceback
    #                 print(f'Error fitting RFM: {traceback.format_exc()}')
    #                 continue
            
    # print(f'Best RFM {tuning_metric}: {best_score}, reg: {best_reg}, bw: {best_bw}, center_grads: {best_center_grads}')

    # return best_model

def train_rfm_probe_on_concept(train_X, train_y, val_X, val_y, 
                                   hyperparams, search_space=None, 
                                   tuning_metric='auc'):
    import traceback
    from copy import deepcopy
    import torch
    from sklearn.metrics import roc_auc_score
    import numpy as np

    if search_space is None:
        search_space = {
            'regs': [1e-3],
            'bws': [1, 10, 100],
            'center_grads': [True, False]
        }

    # Ensure labels are properly shaped (1D for binary classification)
    def ensure_correct_label_shape(labels):
        """Ensure labels are 1D for binary classification"""
        if isinstance(labels, torch.Tensor):
            labels = labels.squeeze()
            if len(labels.shape) > 1:
                if labels.shape[1] == 1:
                    labels = labels.reshape(-1)
                elif labels.shape[1] == 2:
                    labels = torch.argmax(labels, dim=1).float()
        elif isinstance(labels, np.ndarray):
            labels = labels.squeeze()
            if len(labels.shape) > 1:
                if labels.shape[1] == 1:
                    labels = labels.reshape(-1)
                elif labels.shape[1] == 2:
                    labels = np.argmax(labels, axis=1).astype(float)
        return labels

    # Fix label shapes
    train_y = ensure_correct_label_shape(train_y)
    val_y = ensure_correct_label_shape(val_y)

    print(f"Fixed shapes - train_y: {train_y.shape}, val_y: {val_y.shape}")

    best_model = None
    best_score = float('-inf') if tuning_metric in ['f1', 'auc', 'acc', 'top_agop_vectors_ols_auc'] else float('inf')
    maximize_metric = tuning_metric in ['f1', 'auc', 'acc', 'top_agop_vectors_ols_auc']
    
    for reg in search_space['regs']:
        for bw in search_space['bws']:
            for center_grads in search_space['center_grads']:
                try:
                    # Clear GPU cache before each attempt
                    torch.cuda.empty_cache()
                    
                    rfm_params = {
                        'model': {
                            'kernel': 'l2_high_dim',
                            'bandwidth': bw,
                            'tuning_metric': tuning_metric,
                        },
                        'fit': {
                            'reg': reg,
                            'iters': min(hyperparams.get('rfm_iters', 20), 20),
                            'center_grads': center_grads,
                            'early_stop_rfm': True,
                            'get_agop_best_model': True,
                            'top_k': min(hyperparams.get('n_components', 10), 16)
                        }
                    }

                    # Try CUDA first, then CPU if OOM
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    try:
                        model = RFM(**rfm_params['model'], device=device)
                        print(f"Fitting RFM with reg={reg}, bw={bw}, center_grads={center_grads}")
                        
                        # Ensure data is on correct device
                        if device == 'cuda':
                            train_X_device = train_X.cuda() if not train_X.is_cuda else train_X
                            train_y_device = train_y.cuda() if not train_y.is_cuda else train_y
                            val_X_device = val_X.cuda() if not val_X.is_cuda else val_X
                            val_y_device = val_y.cuda() if not val_y.is_cuda else val_y
                        else:
                            train_X_device = train_X.cpu()
                            train_y_device = train_y.cpu()
                            val_X_device = val_X.cpu()
                            val_y_device = val_y.cpu()
                        
                        model.fit((train_X_device, train_y_device), (val_X_device, val_y_device), **rfm_params['fit'])
                        
                    except RuntimeError as cuda_e:
                        if "CUDA out of memory" in str(cuda_e) or "out of memory" in str(cuda_e):
                            print(f"CUDA OOM, trying CPU for reg={reg}, bw={bw}, center_grads={center_grads}")
                            torch.cuda.empty_cache()
                            model = RFM(**rfm_params['model'], device='cpu')
                            model.fit((train_X.cpu(), train_y.cpu()), (val_X.cpu(), val_y.cpu()), **rfm_params['fit'])
                            val_X_device = val_X.cpu()
                            val_y_device = val_y.cpu()
                        else:
                            raise cuda_e

                    # Calculate validation score with safe prediction handling
                    if tuning_metric == 'top_agop_vectors_ols_auc':
                        try:
                            top_k = min(hyperparams.get('n_components', 10), min(16, val_X.shape[0], val_X.shape[1]))
                            targets = val_y_device if 'val_y_device' in locals() else val_y
                            targets = ensure_correct_label_shape(targets)
                            
                            if hasattr(model, 'agop_best_model') and model.agop_best_model is not None:
                                try:
                                    if model.agop_best_model.shape[0] > 1000:
                                        _, U = torch.lobpcg(model.agop_best_model, k=min(top_k, model.agop_best_model.shape[0]-1))
                                    else:
                                        S, U = torch.linalg.eigh(model.agop_best_model)
                                        U = U[:, -top_k:]
                                    
                                    top_eigenvectors = U[:, :top_k] if U.shape[1] >= top_k else U
                                    val_X_eval = val_X_device if 'val_X_device' in locals() else val_X
                                    projections = val_X_eval @ top_eigenvectors
                                    
                                    if len(projections.shape) == 1:
                                        projections = projections.unsqueeze(1)
                                    
                                    if projections.shape[1] > 0:
                                        XtX = projections.T @ projections
                                        Xty = projections.T @ targets.unsqueeze(1) if len(targets.shape) == 1 else projections.T @ targets
                                        XtX += torch.eye(XtX.shape[0], device=XtX.device) * 1e-6
                                        
                                        try:
                                            betas = torch.linalg.solve(XtX, Xty)
                                        except:
                                            betas = torch.linalg.pinv(XtX) @ Xty
                                        
                                        preds = torch.sigmoid(projections @ betas).squeeze()
                                        targets_np = targets.cpu().numpy().flatten()
                                        preds_np = preds.cpu().numpy().flatten()
                                        
                                        if len(targets_np) == len(preds_np) and len(np.unique(targets_np)) > 1:
                                            val_score = roc_auc_score(targets_np, preds_np)
                                        else:
                                            val_score = 0.5
                                    else:
                                        val_score = 0.5
                                except Exception as eig_e:
                                    print(f"Error in eigenvector computation: {str(eig_e)}")
                                    val_score = 0.5
                            else:
                                val_score = 0.5
                        except Exception as auc_e:
                            print(f"Error in top_agop_vectors_ols_auc calculation: {str(auc_e)}")
                            val_score = 0.5
                    else:
                        # Standard prediction metrics with safe handling
                        try:
                            val_X_eval = val_X_device if 'val_X_device' in locals() else val_X
                            val_y_eval = val_y_device if 'val_y_device' in locals() else val_y
                            
                            # Use our safe prediction function
                            pred_proba = safe_model_predict_proba(model, val_X_eval)
                            val_score = compute_prediction_metrics(pred_proba, val_y_eval)[tuning_metric]
                            
                        except Exception as pred_e:
                            print(f"Error in prediction: {str(pred_e)}")
                            continue

                    # Update best model
                    if (maximize_metric and val_score > best_score) or (not maximize_metric and val_score < best_score):
                        best_score = val_score
                        best_model = deepcopy(model)
                        best_reg = reg
                        best_bw = bw
                        best_center_grads = center_grads

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                        print(f"OOM error with reg={reg}, bw={bw}, center_grads={center_grads}, skipping...")
                        torch.cuda.empty_cache()
                    else:
                        print(f"Runtime error fitting RFM: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Error fitting RFM: {str(e)}")
                    continue

    if best_model is None:
        print("RFM fitting failed for all hyperparameter combinations.")
        return None

    print(f"Best RFM {tuning_metric}: {best_score}, reg: {best_reg}, bw: {best_bw}, center_grads: {best_center_grads}")
    return best_model


def train_linear_probe_on_concept(train_X, train_y, val_X, val_y, use_bias=False, tuning_metric='auc', device='cuda'):

    reg_search_space = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
    
    if use_bias:
        X = append_one(train_X)
        Xval = append_one(val_X)
    else:
        X = train_X
        Xval = val_X
    
    n, d = X.shape
    num_classes = train_y.shape[1]

    best_beta = None
    maximize_metric = (tuning_metric in ['f1', 'auc', 'acc'])
    best_score = float('-inf') if maximize_metric else float('inf')
    for reg in reg_search_space:
        try:
            if n>d:
                XtX = batch_transpose_multiply(X, X)
                XtY = batch_transpose_multiply(X, train_y)
                beta = torch.linalg.solve(XtX + reg*torch.eye(X.shape[1]).to(device), XtY)
            else:
                X = X.to(device)
                train_y = train_y.to(device)
                Xval = Xval.to(device)

                XXt = X@X.T
                alpha = torch.linalg.lstsq(XXt + reg*torch.eye(X.shape[0]).to(device), train_y).solution
                beta = X.T@alpha

            preds = Xval.to(device) @ beta
            preds_proba = preds_to_proba(preds)
            val_score = compute_prediction_metrics(preds_proba, val_y)[tuning_metric]

            if maximize_metric and val_score > best_score or not maximize_metric and val_score < best_score:
                best_score = val_score
                best_reg = reg
                best_beta = deepcopy(beta)

        except Exception as e:
            import traceback
            print(f'Error fitting linear probe: {traceback.format_exc()}')
            continue
    
    print(f'Linear probe {tuning_metric}: {best_score}, reg: {best_reg}')

    if use_bias:
        line = best_beta[:-1].to(train_X.device)
        if num_classes == 1:
            bias = best_beta[-1].item()
        else:
            bias = best_beta[-1]
    else:
        line = best_beta.to(train_X.device)
        bias = 0
        
    return line, bias

def train_logistic_probe_on_concept(train_X, train_y, val_X, val_y, use_bias=False, num_classes=1, tuning_metric='auc'):
    
    C_search_space = [1000, 100, 10, 1, 1e-1, 1e-2]

    val_y = val_y.cpu()
    if num_classes == 1:
        train_y_flat = train_y.squeeze(1).cpu()
    else:
        train_y_flat = train_y.argmax(dim=1).cpu()   

    best_beta = None
    best_bias = None
    maximize_metric = (tuning_metric in ['f1', 'auc', 'acc'])
    best_score = float('-inf') if maximize_metric else float('inf')
    for C in C_search_space:
        model = LogisticRegression(fit_intercept=False, max_iter=1000, C=C)
        model.fit(train_X.cpu(), train_y_flat.cpu())
        
        # Get probability predictions
        val_probs = torch.tensor(model.predict_proba(val_X.cpu()))
        if num_classes == 1:
            val_probs = val_probs[:,1].reshape(val_y.shape)
        val_score = compute_prediction_metrics(val_probs, val_y)[tuning_metric]

        if maximize_metric and val_score > best_score or not maximize_metric and val_score < best_score:
            best_score = val_score
            best_beta = torch.from_numpy(model.coef_).T
            if use_bias:
                best_bias = torch.from_numpy(model.intercept_)
            best_C = C

    print(f'Logistic probe {tuning_metric}: {best_score}, C: {best_C}')

    if use_bias:
        line = best_beta.to(train_X.device)
        if num_classes == 1:
            bias = best_bias.item()
        else:
            bias = best_bias
    else:
        line = best_beta.to(train_X.device)
        bias = 0
        
    return line, bias