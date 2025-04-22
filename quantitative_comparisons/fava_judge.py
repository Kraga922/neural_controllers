import argparse
import os
import sys
from pathlib import Path

notebook_path = Path().absolute()
sys.path.append(str(notebook_path.parent))

from utils import load_model
import re
import json
import torch
import pickle
from tqdm import tqdm
import gc

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from abc import ABC, abstractmethod
import direction_utils
from bs4 import BeautifulSoup

_TAGS = ["entity", "relation", "sentence", "invented", "subjective", "unverifiable"]

def remove_deleted_text(text):
    # Use regex to match text between <delete> and </delete> tags
    regex = r'<delete>.*?</delete>'
    
    # Replace all matches with empty string
    # re.DOTALL flag (s) allows matching across multiple lines
    return re.sub(regex, '', text, flags=re.DOTALL)

def remove_empty_tags(html_content):
    # Pattern to match empty tags with optional whitespace between them
    pattern = r'<(\w+)>\s*</\1>'
    
    # Keep removing empty tags until no more changes are made
    prev_content = None
    current_content = html_content
    
    while prev_content != current_content:
        prev_content = current_content
        current_content = re.sub(pattern, '', current_content)
    
    return current_content

def modify(s):
    s = remove_deleted_text(s)
    s = remove_empty_tags(s)

    indicator = [0, 0, 0, 0, 0, 0]
    soup = BeautifulSoup(s, "html.parser")
    s1 = ""
    for t in range(len(_TAGS)):
        indicator[t] = len(soup.find_all(_TAGS[t]))
    # print(soup.find_all(text=True))
    for elem in soup.find_all(text=True):
        if elem.parent.name != "delete":
            s1 += elem
    return s1, int(sum(indicator)>0)

def get_fava_annotated_data():
    # Specify the path to your JSON file
    file_path = './annotations.json'

    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    inputs = []
    labels = []
    for d in data:
        s = d['annotated']
        i, label = modify(s)
        labels.append(label)
        inputs.append(i)
    return inputs, labels


class HallucinationJudge(ABC):
    def __init__(self, judge_prompt):
        self.judge_prompt = judge_prompt
    
    @abstractmethod
    def get_judgement(self, prompt):
        pass
    
    def get_all_predictions(self, inputs):
        """Get predictions for all inputs at once."""
        predictions = []
        probabilities = []  # Store probabilities for AUC calculation
        for input_text in tqdm(inputs):
            prompt = self.judge_prompt.format(statement=input_text)
            print("prompt:", prompt)
            
            judgement, probability = self.get_judgement(prompt)
            predictions.append(int(judgement[0].lower()=='y'))
            probabilities.append(probability)
            print("judgement:", judgement, "probability:", probability)
        
        return torch.tensor(predictions), torch.tensor(probabilities)
    
    def evaluate_split(self, all_predictions, all_probabilities, all_labels, test_indices):
        """Evaluate metrics for a specific split using pre-computed predictions."""
        split_predictions = all_predictions[test_indices]
        split_probabilities = all_probabilities[test_indices]
        split_labels = torch.tensor([all_labels[i] for i in test_indices])
        metrics = direction_utils.compute_classification_metrics(split_predictions, split_labels)
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(split_labels.numpy(), split_probabilities.numpy())
        metrics['auc'] = auc

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
            max_tokens=1,
            temperature=0
        )
        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        yes_prob = None
        no_prob = None
        
        for logprob in logprobs:
            token = logprob.token
            if token == 'Yes':
                yes_prob = logprob.logprob
            elif token == 'No':
                no_prob = logprob.logprob
        
        # If we didn't find exact "Yes"/"No", look for close matches
        if yes_prob is None or no_prob is None:
            for logprob in logprobs:
                token = logprob.token
                if yes_prob is None and ('yes' in token.lower() or 'y' == token.lower()):
                    yes_prob = logprob.logprob
                elif no_prob is None and ('no' in token.lower() or 'n' == token.lower()):
                    no_prob = logprob.logprob
        
        if yes_prob is not None and no_prob is not None:
            # Convert from log probabilities to probabilities
            yes_prob_value = torch.exp(torch.tensor(yes_prob)).item()
            no_prob_value = torch.exp(torch.tensor(no_prob)).item()
            # Normalize probabilities
            total = yes_prob_value + no_prob_value
            yes_prob_norm = yes_prob_value / total if total > 0 else 0.5
            return ('yes' if yes_prob > no_prob else 'no', yes_prob_norm)
        elif yes_prob is not None:
            return ('yes', 1.0)
        elif no_prob is not None:
            return ('no', 0.0)
        else:
            # Fallback to content
            is_yes = 'y' in response.choices[0].message.content.lower()
            return ('yes' if is_yes else 'no', 1.0 if is_yes else 0.0)

class GemmaJudge(HallucinationJudge):
    def __init__(self, judge_prompt):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model('gemma_2_9b_it')
        
    def get_judgement(self, prompt):
        chat = [
            {'role':'user','content':prompt}
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for "yes" and "no"
        yes_token_id = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode('No', add_special_tokens=False)[0]
        
        # We'll use the model's generate method with return_dict_in_generate=True
        # to get the full scores (probabilities) for each possible next token
        with torch.no_grad():
            # Set up generation parameters that match exactly what would be used during sampling
            gen_kwargs = {
                "max_new_tokens": 1,  # We only care about the first token
                "do_sample": False,   # No sampling, we just want the probabilities
                "return_dict_in_generate": True,
                "output_scores": True,
                "temperature": 1.0,   # No temperature scaling
            }
            
            # Generate a token and get the scores
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Extract the scores (logits) for the first generated token
            token_scores = outputs.scores[0][0]  # [batch, sequence_position, vocab_size]
            
            # Apply softmax the same way the model would during sampling
            token_probs = torch.nn.functional.softmax(token_scores, dim=-1)
            
            # Get the probabilities for "yes" and "no" tokens
            yes_prob = token_probs[yes_token_id].item()
            no_prob = token_probs[no_token_id].item()
            
            # Normalize the probabilities so they sum to 1
            total_prob = yes_prob + no_prob
            yes_prob_norm = yes_prob / total_prob if total_prob > 0 else 0.5
            
            # Make the decision based on these probabilities
            return ('yes' if yes_prob > no_prob else 'no', yes_prob_norm)
    
class LlamaJudge(HallucinationJudge):
    def __init__(self, judge_prompt):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model('llama_3_8b_it')
        
    def get_judgement(self, prompt):
        chat = [
            {'role':'system', 'content':'You are a helpful assistant who follows instructions exactly.'},
            {'role':'user', 'content':prompt}
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for "yes" and "no"
        yes_token_id = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode('No', add_special_tokens=False)[0]
        
        # We'll use the model's generate method with return_dict_in_generate=True
        # to get the full scores (probabilities) for each possible next token
        with torch.no_grad():
            # Set up generation parameters that match exactly what would be used during sampling
            gen_kwargs = {
                "max_new_tokens": 1,  # We only care about the first token
                "do_sample": False,   # No sampling, we just want the probabilities
                "return_dict_in_generate": True,
                "output_scores": True,
                "temperature": 1.0,   # No temperature scaling
            }
            
            # Generate a token and get the scores
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Extract the scores (logits) for the first generated token
            token_scores = outputs.scores[0][0]  # [batch, sequence_position, vocab_size]
            
            # Apply softmax the same way the model would during sampling
            token_probs = torch.nn.functional.softmax(token_scores, dim=-1)
            
            # Get the probabilities for "yes" and "no" tokens
            yes_prob = token_probs[yes_token_id].item()
            no_prob = token_probs[no_token_id].item()
            
            # Normalize the probabilities so they sum to 1
            total_prob = yes_prob + no_prob
            yes_prob_norm = yes_prob / total_prob if total_prob > 0 else 0.5
            
            # Make the decision based on these probabilities
            return ('yes' if yes_prob > no_prob else 'no', yes_prob_norm)

def save_predictions(predictions, probabilities, judge_type, judge_model):
    """Save all predictions to a file."""
    out_name = f'./fava_annotated_results/{judge_type}_{judge_model}_all_predictions.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump({'predictions': predictions, 'probabilities': probabilities}, f)

def load_predictions(judge_type, judge_model):
    """Load predictions from file if they exist."""
    out_name = f'./fava_annotated_results/{judge_type}_{judge_model}_all_predictions.pkl'
    if os.path.exists(out_name):
        with open(out_name, 'rb') as f:
            data = pickle.load(f)
            return data['predictions'], data['probabilities']
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3_8b_it')
    parser.add_argument('--n_components', type=int, default=3)
    parser.add_argument('--judge_type', type=str, choices=['openai', 'llama', 'gemma'], default='llama')
    parser.add_argument('--judge_model', type=str, default='gpt-4o')
    parser.add_argument('--n_seeds', type=int, default=20)
    parser.add_argument('--n_train', type=int, default=0)
    parser.add_argument('--n_val', type=int, default=360)
    args = parser.parse_args()
    
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    inputs, labels = get_fava_annotated_data()
    out_name = f'./fava_annotated_results/splits_ntrain_0_nval_360_ntotal_460_nseeds_20.pkl'
    with open(out_name, 'rb') as f:
        splits = pickle.load(f)
    
    judge_prompt = "Consider hallucinations of the following types:\n"
    judge_prompt += (
        "(1a) Entity : Contradictory entity errors are a sub-category within Type 1, "
        "where an entity in a statement is incorrect and changing that single entity "
        "can make the entire sentence factually correct.\n"
        
        "(1b) Relation : Contradictory relation errors are another sub-category within "
        "contradictory statements where a semantic relationship (e.g., verbs, prepositions, "
        "or adjectives) in a statement is incorrect.\n"
        
        "(1c) Sentence : Contradictory sentence errors refer to cases where a full statement "
        "entirely contradicts relevant evidence from the web, and cannot be solved via "
        "phrase-level edits.\n"
        
        "(2) Invented : Invented errors refer to statements where the LM generates an entirely "
        "fabricated entity that doesn't exist based on world knowledge. Fictional entities in "
        "creative work aren't included.\n"
        
        "(3) Subjective : Subjective errors refer to expressions about existing entities that "
        "lack universal validity. These statements often do not contain facts and are influenced "
        "by personal beliefs or opinions.\n"
        
        "(4) Unverifiable : These are statements where the LM output contains facts, but no "
        "retrieved.\n\n"
    )
    judge_prompt += (
        'Based on the above definition, does the following statement contain a hallucination? '
        'Simply state yes or no.\nStatement: {statement}'
    )
    
    if args.judge_type == 'openai':
        judge = OpenAIJudge(judge_prompt, args.judge_model)
    elif args.judge_type == 'llama':
        judge = LlamaJudge(judge_prompt)
    elif args.judge_type == 'gemma':
        judge = GemmaJudge(judge_prompt)

    # Try to load predictions or generate new ones
    all_predictions, all_probabilities = load_predictions(args.judge_type, args.judge_model)
    if all_predictions is None:
        all_predictions, all_probabilities = judge.get_all_predictions(inputs)
        save_predictions(all_predictions, all_probabilities, args.judge_type, args.judge_model)
    
    # Iterate over seeds using pre-computed predictions
    for seed in tqdm(range(args.n_seeds)):
        split = splits[seed]
        
        metrics = judge.evaluate_split(all_predictions, all_probabilities, labels, split['test_indices'])
        print(f"Seed {seed} metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        out_name = f'./fava_annotated_results/{args.judge_type}_{args.judge_model}_seed_{seed}_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(metrics, f)
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()