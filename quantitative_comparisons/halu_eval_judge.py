import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

NEURAL_CONTROLLERS_DIR = os.environ.get('NEURAL_CONTROLLERS_DIR', str(Path(__file__).parent.parent))
RESULTS_DIR = f'{NEURAL_CONTROLLERS_DIR}/results'

from utils import load_model

import torch
import pickle
from direction_utils import compute_prediction_metrics
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from abc import ABC, abstractmethod
from utils import load_model
from sklearn.metrics import roc_auc_score

def read_hallucination_prompts_from_lines(lines):
    import re
    
    dicts = []
    for line in lines:
        line = line[1:-1]
        x = re.findall('".*?"', line)
        
        prompt = {}
        prompt['knowledge'] = x[1].strip('"')
        prompt['question'] = x[3].strip('"')
        prompt['answer'] = x[5].strip('"')
        prompt['hallucination'] = x[7].strip('"')
        dicts.append(prompt)
    return dicts

def clean(prompts):
    new_prompts = []
    for p in prompts:
        if p['question'] == 'question':
            continue
        new_prompts.append(p)
    return new_prompts

def get_halu_eval_data(hal_type):
    if hal_type=='qa':
        data_path = f"{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval/qa_data.txt"
        template = 'Yes or no, is the answer to the following question factual?\n\nQ: {question}\n\nA: {answer}'

        with open(data_path, 'r') as f:
            lines = f.readlines()
            raw_prompts = read_hallucination_prompts_from_lines(lines)
            n = len(raw_prompts)
            # Only use evaluation data (second half)
            raw_prompts = raw_prompts[int(n//2):]

        prompts = clean(raw_prompts)
        inputs = []
        labels = []
        for prompt in prompts:
            x_pos = template.format(question=prompt['question'], answer=prompt['answer'])
            x_neg = template.format(question=prompt['question'], answer=prompt['hallucination'])
            inputs.append(x_pos)
            inputs.append(x_neg)
            labels += [0,1]
            
    elif hal_type=='general':
        data_path = f"{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval/general_data.txt"
        template = 'Is the response to the following query factual? Simply state yes or no.\n\nQuery: {query}\n\nResponse: {response}'
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
            raw_prompts = read_hallucination_prompts_from_lines(lines)
            prompts = clean(raw_prompts)
            
        inputs = []
        labels = []
        for prompt in prompts:
            x = template.format(query=prompt['question'], response=prompt['answer'])
            inputs.append(x)
            # Flip the label logic to match the other code
            label = 0 if prompt['hallucination']=='no' else 1
            labels.append(label)
        
    return inputs, labels

class HallucinationJudge(ABC):
    def __init__(self, judge_prompt):
        self.judge_prompt = judge_prompt
    
    @abstractmethod
    def get_judgement(self, prompt):
        """Get a single judgement with probability for a prompt."""
        pass
    
    def get_all_predictions(self, inputs):
        """Get predictions for all inputs at once."""
        predictions = []
        probabilities = []  # Store probabilities for AUC calculation
        for input_text in tqdm(inputs):
            prediction, probability = self.get_judgement(input_text)
            predictions.append(prediction)
            probabilities.append(probability)
        
        return torch.tensor(predictions), torch.tensor(probabilities)
    
    def evaluate_split(self, all_predictions, all_probabilities, all_labels, test_indices):
        """Evaluate metrics for a specific split using pre-computed predictions."""
        split_probabilities = all_probabilities[test_indices]
        split_labels = torch.tensor([all_labels[i] for i in test_indices])
        
        split_probabilities = torch.tensor(split_probabilities).reshape(-1, 1)
        split_labels = torch.tensor(split_labels).reshape(-1, 1)
        metrics = compute_prediction_metrics(split_probabilities, split_labels)
        
        # Calculate AUC
        auc = roc_auc_score(split_labels.numpy(), split_probabilities.numpy())
        metrics['auc'] = auc
        
        return metrics
        
    def evaluate_inputs(self, test_inputs, test_labels, splits):
        # First try to load precomputed predictions
        judge_type = self.__class__.__name__.replace('Judge', '').lower()
        judge_model = getattr(self, 'judge_model', None)
        
        all_predictions, all_probabilities = load_predictions(judge_type, judge_model)
        
        if all_predictions is None:
            # Get all judgements at once for efficiency
            all_predictions, all_probabilities = self.get_all_predictions(test_inputs)
            save_predictions(all_predictions, all_probabilities, judge_type, judge_model)
        
        results = []
        for seed in range(len(splits)):
            split = splits[seed]
            
            # Evaluate using the pre-computed predictions
            test_metrics = self.evaluate_split(all_predictions, all_probabilities, test_labels, split['test_indices'])
            
            results.append({
                'seed': seed,
                'test_metrics': test_metrics,
            })
            
        return results

class OpenAIJudge(HallucinationJudge):
    def __init__(self, judge_prompt, judge_model):
        super().__init__(judge_prompt)
        self.judge_model = judge_model
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    @retry(
        stop=stop_after_attempt(12),
        wait=wait_exponential(min=1, max=1024),
    )
    def get_judgement(self, prompt):
        full_prompt = self.judge_prompt.format(statement=prompt)
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                {"role": "user", "content": full_prompt}
            ],
            logprobs=True,
            top_logprobs=20,
            max_tokens=5,
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
            no_prob_norm = no_prob_value / total if total > 0 else 0.5
            return (no_prob > yes_prob, no_prob_norm)
        elif yes_prob is not None:
            return (0, 0.0)
        elif no_prob is not None:
            return (1, 1.0)
        else:
            # Fallback to content
            is_no = 'n' in response.choices[0].message.content.lower()
            return (is_no, is_no)

class LlamaJudge(HallucinationJudge):
    def __init__(self, judge_prompt, judge_model=None):
        super().__init__(judge_prompt)
        self.judge_model = judge_model
        self.model, self.tokenizer = load_model(self.judge_model)
        
    def get_judgement(self, prompt):
        full_prompt = self.judge_prompt.format(statement=prompt)
        chat = [
            {'role':'system', 'content':'You are a helpful assistant who follows instructions exactly.'},
            {'role':'user', 'content':full_prompt}
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for "yes" and "no"
        yes_token_id = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode('No', add_special_tokens=False)[0]
        
        # We'll use the model's generate method with return_dict_in_generate=True
        # to get the full scores (probabilities) for each possible next token
        with torch.no_grad():
            # Set up generation parameters for probability calculation
            gen_kwargs = {
                "max_new_tokens": 1,  # We only care about the first token
                "do_sample": False,   # No sampling, we just want the probabilities
                "return_dict_in_generate": True,
                "output_scores": True,
                "temperature": 0.0,   # No temperature scaling
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
            no_prob_norm = no_prob / total_prob if total_prob > 0 else 0.5
            
            # Make the decision based on these probabilities
            return (no_prob > yes_prob, no_prob_norm)

class GemmaJudge(HallucinationJudge):
    def __init__(self, judge_prompt, judge_model=None):
        super().__init__(judge_prompt)
        self.judge_model = judge_model
        self.model, self.tokenizer = load_model(self.judge_model)
        
    def get_judgement(self, prompt):
        full_prompt = self.judge_prompt.format(statement=prompt)
        chat = [
            {'role':'user', 'content':full_prompt}
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for "yes" and "no"
        yes_token_id = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode('No', add_special_tokens=False)[0]
        
        # We'll use the model's generate method with return_dict_in_generate=True
        # to get the full scores (probabilities) for each possible next token
        with torch.no_grad():
            # Set up generation parameters for probability calculation
            gen_kwargs = {
                "max_new_tokens": 1,  # We only care about the first token
                "do_sample": False,   # No sampling, we just want the probabilities
                "return_dict_in_generate": True,
                "output_scores": True,
                "temperature": 0.0,   # No temperature scaling
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
            no_prob_norm = no_prob / total_prob if total_prob > 0 else 0.5
            
            # Make the decision based on these probabilities
            return (no_prob > yes_prob, no_prob_norm)

def save_predictions(predictions, probabilities, judge_type, judge_model):
    """Save all predictions to a file."""
    os.makedirs(f'{RESULTS_DIR}/halu_eval_results', exist_ok=True)
    out_name = f'{RESULTS_DIR}/halu_eval_results/{judge_type}_{judge_model}_all_predictions.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump({'predictions': predictions, 'probabilities': probabilities}, f)

def load_predictions(judge_type, judge_model):
    """Load predictions from file if they exist."""
    out_name = f'{RESULTS_DIR}/halu_eval_results/{judge_type}_{judge_model}_all_predictions.pkl'
    if os.path.exists(out_name):
        with open(out_name, 'rb') as f:
            data = pickle.load(f)
            return data['predictions'], data['probabilities']
    return None, None
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--hal_type', type=str, default='qa')
    parser.add_argument('--judge_type', type=str, default='llama', choices=['openai', 'llama', 'gemma'])
    parser.add_argument('--judge_model', type=str, default='llama_3.3_70b_4bit_it', choices=['gpt-4o', 'llama_3.3_70b_4bit_it'])
    args = parser.parse_args()
    
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    inputs, labels = get_halu_eval_data(args.hal_type)
    
    # Get the same splits as used in the original code
    if args.hal_type == 'qa':
        out_name = f'{RESULTS_DIR}/halu_eval_results/qa_test_splits_nval_3997_ntotal_7994_nseeds_{args.n_seeds}.pkl'
    else:
        out_name = f'{RESULTS_DIR}/halu_eval_results/general_test_splits_nval_2253_ntotal_4507_nseeds_{args.n_seeds}.pkl'
    
    os.makedirs(f'{RESULTS_DIR}/halu_eval_results', exist_ok=True)
    with open(out_name, 'rb') as f:
        splits = pickle.load(f)
    
    judge_prompt = '{statement}'
    if args.judge_type == 'openai':
        judge = OpenAIJudge(judge_prompt, args.judge_model)
    elif args.judge_type == 'llama':
        judge = LlamaJudge(judge_prompt, args.judge_model)
    elif args.judge_type == 'gemma':
        judge = GemmaJudge(judge_prompt, args.judge_model)
    
    # Evaluate and get results for all seeds
    results = judge.evaluate_inputs(inputs, labels, splits)
    
    # Print average metrics across seeds
    print("\nAverage metrics across all seeds:")
    avg_metrics = {}
    for metric_name in results[0]['test_metrics'].keys():
        avg_metrics[metric_name] = sum(result['test_metrics'][metric_name] for result in results) / len(results)
        print(f"{metric_name}: {avg_metrics[metric_name]:.4f}")
    
    # Save results for each seed
    for result in results:
        seed = result['seed']
        test_metrics = result['test_metrics']
        test_out_name = f'{RESULTS_DIR}/halu_eval_results/{args.judge_type}_{args.judge_model}_{args.hal_type}_seed_{seed}_metrics.pkl'
        with open(test_out_name, 'wb') as f:
            pickle.dump(test_metrics, f)

if __name__ == '__main__':
    main()