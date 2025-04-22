import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

NEURAL_CONTROLLERS_DIR = os.environ.get('NEURAL_CONTROLLERS_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = f'{NEURAL_CONTROLLERS_DIR}/results'

from utils import load_model

import json
import torch
import pickle
from tqdm import tqdm
import random
from openai import OpenAI
from abc import ABC, abstractmethod

import direction_utils  
random.seed(0)

def read_json_to_list(file_path):
    """
    Reads a JSON file and returns its contents as a list of dictionaries.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List of dictionaries loaded from the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if isinstance(data, list):
        return data
    else:
        raise ValueError("JSON content is not a list.")


TYPES = ['confused / erroneous queries', 'inappropriate content', 'complex reasoning', 
         'out-of-scope information', 'beyond-modality interaction', 'other types']

def get_multiclass_halu_eval_wild_data():
    data_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval_wild/HaluEval_Wild_6types.json'
    entries = read_json_to_list(data_path)
    
    # Get unique classes from TYPE_MAP
    classes = TYPES
    num_classes = len(classes)

    print("classes", classes)
    
    inputs = []
    labels = []
    ohe_labels = []
    
    for entry in entries:
        query = entry['query']
        qtype = entry['query_type']
        inputs.append(query)
        
        # Create one-hot encoded label
        labels.append(qtype)

        label = [0] * num_classes
        class_idx = classes.index(qtype)
        label[class_idx] = 1
        ohe_labels.append(torch.tensor(label))
    
    ohe_labels = torch.stack(ohe_labels).reshape(-1, num_classes).cuda().float()

    return inputs, ohe_labels, labels   

class HallucinationJudge(ABC):
    def __init__(self, judge_prompt):
        self.judge_prompt = judge_prompt
    
    @abstractmethod
    def get_judgement(self, prompt):
        pass
    
    def get_all_predictions(self, inputs, num_classes=6):
        """Get predictions for all inputs at once."""
        predictions = []
        probabilities = []
        for input_text in tqdm(inputs):
            prompt = self.judge_prompt.format(query=input_text)
            print("prompt:", prompt)
            
            judgement, probs = self.get_judgement(prompt)
            print("judgement:", judgement, "probabilities:", probs)
            
            try:
                pred = int(judgement[0])-1
            except:
                print("Error:", judgement)
                pred = 0  # Default to first class if parsing fails
                
            pred_ohe = torch.zeros(num_classes).cuda()
            pred_ohe[pred] = 1
            predictions.append(pred_ohe)
            probabilities.append(torch.tensor(probs).cuda())
        
        return torch.stack(predictions), torch.stack(probabilities)
    
    def evaluate_split(self, all_predictions, all_probabilities, all_labels, test_indices):
        """Evaluate metrics for a specific split using pre-computed predictions."""
        split_probabilities = all_probabilities[test_indices]
        split_labels = all_labels[test_indices]
        
        metrics = direction_utils.compute_prediction_metrics(split_probabilities, split_labels)
        
        # Additional metrics based on probabilities could be added here
        # For example, cross entropy or other probability-based metrics
        
        return metrics

class OpenAIJudge(HallucinationJudge):
    def __init__(self, judge_prompt, judge_model):
        super().__init__(judge_prompt)
        self.judge_model = judge_model
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    def get_judgement(self, prompt):
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                {"role": "user", "content": prompt}
            ],
            logprobs=True,
            top_logprobs=20,
            max_tokens=5,
            temperature=0
        )
        
        # Process the top token
        judgment = response.choices[0].message.content.strip()
        
        # Extract logprobs for each class
        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        
        # Initialize probabilities for each class
        class_probs = [0.0] * len(TYPES)
        
        # Parse logprobs to find digit tokens
        for logprob in logprobs:
            token = logprob.token.strip()
            if token in ["1", "2", "3", "4", "5", "6"]:
                class_idx = int(token) - 1
                if 0 <= class_idx < len(class_probs):
                    # Convert from log probability to actual probability
                    class_probs[class_idx] = torch.exp(torch.tensor(logprob.logprob)).item()
        
        # Normalize probabilities to sum to 1
        total_prob = sum(class_probs)
        if total_prob > 0:
            class_probs = [p / total_prob for p in class_probs]
        else:
            # If no valid class probabilities found, assign probability 1.0 to parsed class
            try:
                pred_class = int(judgment[0]) - 1
                if 0 <= pred_class < len(class_probs):
                    class_probs = [0.0] * len(TYPES)
                    class_probs[pred_class] = 1.0
            except:
                # Default to uniform distribution if parsing fails
                class_probs = [1.0/len(TYPES)] * len(TYPES)
                
        return judgment, class_probs

class GemmaJudge(HallucinationJudge):
    def __init__(self, judge_prompt, judge_model):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model(judge_model)
        
    def get_judgement(self, prompt):
        chat = [
            {'role':'user',
             'content':prompt
            }
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for digits 1-6
        digit_ids = []
        for i in range(1, 7):
            token_id = self.tokenizer.encode(str(i), add_special_tokens=False)[0]
            digit_ids.append(token_id)
        
        with torch.no_grad():
            # Generate with return_dict_in_generate and output_scores to get logits
            gen_kwargs = {
                "max_new_tokens": 5,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
                "temperature": 0.0,
            }
            
            # Generate output for judgment
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Extract the scores (logits) for the first generated token
            token_scores = outputs.scores[0][0]
            
            # Apply softmax to get probabilities
            token_probs = torch.nn.functional.softmax(token_scores, dim=-1)
            
            # Get probabilities for digits 1-6
            class_probs = [0.0] * len(TYPES)
            for i, token_id in enumerate(digit_ids):
                class_probs[i] = token_probs[token_id].item()
            
            # Normalize probabilities
            total_prob = sum(class_probs)
            if total_prob > 0:
                class_probs = [p / total_prob for p in class_probs]
            
            # Find the most likely class based on probabilities
            most_likely_class = str(class_probs.index(max(class_probs)) + 1)
        

        return most_likely_class, class_probs
    
class LlamaJudge(HallucinationJudge):
    def __init__(self, judge_prompt, judge_model):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model(judge_model)
        
    def get_judgement(self, prompt):
        chat = [
            {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
            {'role':'user', 'content': prompt }
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for digits 1-6
        digit_ids = []
        for i in range(1, 7):
            token_id = self.tokenizer.encode(str(i), add_special_tokens=False)[0]
            digit_ids.append(token_id)
        
        with torch.no_grad():
            # Generate with return_dict_in_generate and output_scores to get logits
            gen_kwargs = {
                "max_new_tokens": 5,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
                "temperature": 0.0,
            }
            
            # Generate output for judgment
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Extract the scores (logits) for the first generated token
            token_scores = outputs.scores[0][0]
            
            # Apply softmax to get probabilities
            token_probs = torch.nn.functional.softmax(token_scores, dim=-1)
            
            # Get probabilities for digits 1-6
            class_probs = [0.0] * len(TYPES)
            for i, token_id in enumerate(digit_ids):
                class_probs[i] = token_probs[token_id].item()
            
            # Normalize probabilities
            total_prob = sum(class_probs)
            if total_prob > 0:
                class_probs = [p / total_prob for p in class_probs]
            
            # Find the most likely class based on probabilities
            most_likely_class = str(class_probs.index(max(class_probs)) + 1)
        
        return most_likely_class, class_probs

def save_predictions(predictions, probabilities, judge_type, judge_model):
    """Save all predictions to a file."""
    os.makedirs(f'{RESULTS_DIR}/halu_eval_wild_results', exist_ok=True)
    out_name = f'{RESULTS_DIR}/halu_eval_wild_results/multiclass_{judge_type}_{judge_model}_all_predictions.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump({'predictions': predictions, 'probabilities': probabilities}, f)

def load_predictions(judge_type, judge_model):
    """Load predictions from file if they exist."""
    out_name = f'{RESULTS_DIR}/halu_eval_wild_results/multiclass_{judge_type}_{judge_model}_all_predictions.pkl'
    if os.path.exists(out_name):
        with open(out_name, 'rb') as f:
            data = pickle.load(f)
            return data['predictions'], data['probabilities']
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--n_train', type=int, default=300)
    parser.add_argument('--n_val', type=int, default=250)
    parser.add_argument('--judge_type', type=str, default='llama', choices=['openai', 'llama', 'gemma'])
    parser.add_argument('--judge_model', type=str, default='llama_3.3_70b_4bit_it', choices=['gpt-4o', 'llama_3.3_70b_4bit_it'])
    args = parser.parse_args()
    
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")

    inputs, labels, qtypes = get_multiclass_halu_eval_wild_data()

    splits_dir = f'{RESULTS_DIR}/halu_eval_wild_results'
    os.makedirs(splits_dir, exist_ok=True)
    out_name = f'{splits_dir}/splits_ntrain_{args.n_train}_nval_{args.n_val}_ntotal_600_nseeds_{args.n_seeds}.pkl'
    with open(out_name, 'rb') as f:
        splits = pickle.load(f)

    judge_prompt = "Queries that induce hallucinations consist of the following six types. "
    judge_prompt += "(1) Confused / Erroneous queries: Queries that contain errors in the entity, relation, or sentence. "
    judge_prompt += "(2) Inappropriate content: Queries that request inappropriate content. "
    judge_prompt += "(3) Complex reasoning: Queries that require complex reasoning. "
    judge_prompt += "(4) Out-of-scope information: Queries that ask for information out-of-scope for the LLM. "
    judge_prompt += "(5) Beyond-modality interaction: Queries that require modalities beyond the abilities of the language model being queried. "
    judge_prompt += "(6) Other types: Queries that are not out-of-scope, do not require complex reasoning, are not beyond-modality, are not inappropriate, and are not confused or erroneous. " 
    judge_prompt += "Based on the above definitions, which single category does the following query fall into? Respond just with a number between 1 and 6. "
    judge_prompt += "For example, your response would be just 'N.' if the query belongs to category N.\n\n"
    judge_prompt += "Query: {query}"

    if args.judge_type == 'openai':
        judge = OpenAIJudge(judge_prompt, args.judge_model)
    elif args.judge_type == 'llama':
        judge = LlamaJudge(judge_prompt, args.judge_model)
    elif args.judge_type == 'gemma':
        judge = GemmaJudge(judge_prompt, args.judge_model)
 
    
    # Try to load predictions or generate new ones
    all_predictions, all_probabilities = load_predictions(args.judge_type, args.judge_model)
    if all_predictions is None:
        all_predictions, all_probabilities = judge.get_all_predictions(inputs)
        save_predictions(all_predictions, all_probabilities, args.judge_type, args.judge_model)
    
    # Collect metrics across all seeds
    all_seed_metrics = []
    
    # Evaluate each seed using the pre-computed predictions
    for seed in tqdm(range(args.n_seeds)):
        split = splits[seed]
        metrics = judge.evaluate_split(all_predictions, all_probabilities, labels, split['test_indices'])
        
        print(f"Seed {seed} metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        out_dir = f'{RESULTS_DIR}/halu_eval_wild_results'
        os.makedirs(out_dir, exist_ok=True)
        out_name = f'{out_dir}/multiclass_{args.judge_type}_{args.judge_model}_seed_{seed}_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(metrics, f)
            
        all_seed_metrics.append(metrics)
        
    # Calculate and print average metrics across all seeds
    print("\nAverage metrics across all seeds:")
    avg_metrics = {}
    for metric_name in all_seed_metrics[0].keys():
        avg_metrics[metric_name] = sum(seed_metric[metric_name] for seed_metric in all_seed_metrics) / len(all_seed_metrics)
        print(f"{metric_name}: {avg_metrics[metric_name]:.4f}")

if __name__ == '__main__':
    main()