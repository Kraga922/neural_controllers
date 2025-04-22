import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_model

import json
import torch
import pickle
from tqdm import tqdm
import gc
import random
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
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
    data_path = '../data/hallucinations/halu_eval_wild/HaluEval_Wild_6types.json'
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
                
            pred_ohe = torch.zeros(num_classes)
            pred_ohe[pred] = 1
            predictions.append(pred_ohe)
            probabilities.append(torch.tensor(probs))
        
        return torch.stack(predictions).cuda(), torch.stack(probabilities).cuda()
    
    def evaluate_split(self, all_predictions, all_probabilities, all_labels, test_indices):
        """Evaluate metrics for a specific split using pre-computed predictions."""
        split_predictions = all_predictions[test_indices]
        split_probabilities = all_probabilities[test_indices]
        split_labels = all_labels[test_indices]
        
        metrics = direction_utils.compute_classification_metrics(split_predictions, split_labels)
        
        # Additional metrics based on probabilities could be added here
        # For example, cross entropy or other probability-based metrics
        
        return metrics

class OpenAIJudge(HallucinationJudge):
    def __init__(self, judge_prompt, model_name):
        super().__init__(judge_prompt)
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    @retry(
        stop=stop_after_attempt(12),
        wait=wait_exponential(min=1, max=1024), 
    )
    def get_judgement(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                {"role": "user", "content": prompt}
            ],
            logprobs=True,
            top_logprobs=50,
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
    def __init__(self, judge_prompt, model_path=None):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model('gemma_2_9b_it')
        
    def get_judgement(self, prompt):
        chat = [
            {'role':'user',
             'content':prompt
            }
        ]
        assistant_tag = '<bos><start_of_turn>model\n\n'
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        wrapped_prompt += assistant_tag
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for digits 1-6
        digit_ids = []
        class_probs = [0.0] * len(TYPES)
        for i in range(1, 7):
            token_id = self.tokenizer.encode(str(i), add_special_tokens=False)[0]
            digit_ids.append(token_id)
        
        with torch.no_grad():
            # Generate output for judgment
            response = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )
            
            # Get logits for first token prediction
            outputs = self.model(inputs.input_ids)
            logits = outputs.logits[:, -1, :]
            
            # Get probabilities for digits 1-6
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            for i, token_id in enumerate(digit_ids):
                class_probs[i] = probs[token_id].item()
            
            # Normalize probabilities
            total_prob = sum(class_probs)
            if total_prob > 0:
                class_probs = [p / total_prob for p in class_probs]
        
        # Decode response
        text_response = self.tokenizer.decode(response[0])
        text_response = text_response[text_response.find(assistant_tag)+len(assistant_tag):]
        text_response = text_response.strip().strip('\n').replace('*','')
        
        return text_response, class_probs
    
class LlamaJudge(HallucinationJudge):
    def __init__(self, judge_prompt, model_path=None):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model('llama_3_8b_it')
        
    def get_judgement(self, prompt):
        chat = [
            {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
            {'role':'user', 'content': prompt }
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for digits 1-6
        digit_ids = []
        class_probs = [0.0] * len(TYPES)
        for i in range(1, 7):
            token_id = self.tokenizer.encode(str(i), add_special_tokens=False)[0]
            digit_ids.append(token_id)
        
        with torch.no_grad():
            # Generate output for judgment
            response = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )
            
            # Get logits for first token prediction
            outputs = self.model(inputs.input_ids)
            logits = outputs.logits[:, -1, :]
            
            # Get probabilities for digits 1-6
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            for i, token_id in enumerate(digit_ids):
                class_probs[i] = probs[token_id].item()
            
            # Normalize probabilities
            total_prob = sum(class_probs)
            if total_prob > 0:
                class_probs = [p / total_prob for p in class_probs]
        
        # Decode response
        text_response = self.tokenizer.decode(response[0])
        assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
        text_response = text_response[text_response.find(assistant_tag)+len(assistant_tag):]
        
        return text_response.strip(), class_probs

def save_predictions(predictions, probabilities, judge_type, judge_model):
    """Save all predictions to a file."""
    out_name = f'./halu_eval_wild_results/multiclass_{judge_type}_{judge_model}_all_predictions.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump({'predictions': predictions, 'probabilities': probabilities}, f)

def load_predictions(judge_type, judge_model):
    """Load predictions from file if they exist."""
    out_name = f'./halu_eval_wild_results/multiclass_{judge_type}_{judge_model}_all_predictions.pkl'
    if os.path.exists(out_name):
        with open(out_name, 'rb') as f:
            data = pickle.load(f)
            return data['predictions'], data['probabilities']
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=20)
    parser.add_argument('--n_train', type=int, default=200)
    parser.add_argument('--n_val', type=int, default=350)
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--judge_type', type=str, choices=['openai', 'llama', 'gemma'], default='llama')
    parser.add_argument('--judge_model', type=str, default='gpt-4o')
    args = parser.parse_args()
    
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")

    inputs, labels, qtypes = get_multiclass_halu_eval_wild_data()

    out_name = f'./halu_eval_wild_results/splits_ntrain_300_nval_250_ntotal_600_nseeds_20.pkl'
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
        judge = LlamaJudge(judge_prompt)
    elif args.judge_type == 'gemma':
        judge = GemmaJudge(judge_prompt)
 
    def get_counts(labels):
        counts = {}
        for label in labels:
            class_idx = label.argmax().item()
            counts[class_idx] = counts.get(class_idx, 0) + 1
        return counts
 
    print('counts: ', get_counts(labels))
    
    # Try to load predictions or generate new ones
    all_predictions, all_probabilities = load_predictions(args.judge_type, args.judge_model)
    if all_predictions is None:
        all_predictions, all_probabilities = judge.get_all_predictions(inputs)
        save_predictions(all_predictions, all_probabilities, args.judge_type, args.judge_model)
    
    preds = all_predictions.argmax(dim=1)
    targets = labels.argmax(dim=1)
    print('Accuracy: ', (preds == targets).float().mean().item()*100)
    
    # Evaluate each seed using the pre-computed predictions
    for seed in tqdm(range(args.n_seeds)):
        split = splits[seed]
        metrics = judge.evaluate_split(all_predictions, all_probabilities, labels, split['test_indices'])
        
        print(f"Seed {seed} metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        out_name = f'./halu_eval_wild_results/multiclass_{args.judge_type}_{args.judge_model}_seed_{seed}_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(metrics, f)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()