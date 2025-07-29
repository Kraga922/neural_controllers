# import pickle
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from torch.nn.functional import cosine_similarity
# import string

# # Load direction vector from pkl
# # with open('directions/rfm_harmful_llama_3_8b_it_detector.pkl', 'rb') as f:
# #     directions = pickle.load(f)

# with open('directions/rfm_harmful_llama_3_8b_it.pkl', 'rb') as f:
#     directions = pickle.load(f)

# # print("Type of object loaded:", type(directions))
# # print("Keys in pkl_data:", list(directions.keys()))



# dir_at_last_layer = directions[-1]
# direction_vector = dir_at_last_layer[0].detach().to('cuda')  

# # Load tokenizer and model
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name).eval().to('cuda')  # Move model to GPU

# # Get embedding matrix and move to GPU
# embedding_matrix = model.get_input_embeddings().weight.detach().to('cuda')

# # Normalize direction and embeddings
# direction_norm = direction_vector / direction_vector.norm()
# embedding_norms = embedding_matrix / embedding_matrix.norm(dim=1, keepdim=True)

# # Compute cosine similarity
# projections = torch.matmul(embedding_norms, direction_norm)  # (vocab_size,)

# # Get top-k most similar tokens
# topk = 50
# top_indices = torch.topk(projections, topk).indices
# top_tokens = [tokenizer.decode([i]) for i in top_indices]

# # print("Top matching tokens:", top_tokens)

# prompt = " ".join(top_tokens)
# # print("Generated Prompt:", prompt)


# def is_clean_token(token):
#     token = token.strip()
#     return (
#         len(token) > 1 and
#         all(c in string.printable for c in token) and
#         any(c.isalpha() for c in token)
#     )

# filtered_tokens = [tok for tok in top_tokens if is_clean_token(tok)]
# # print("Filtered Tokens:", filtered_tokens)
# prompt = " ".join(filtered_tokens)
# print("Filtered Prompt:", prompt)

# ###another version of decode_pkl.py
# # This script optimizes a soft prompt to align with a specified direction vector from a .pkl file.
# # It uses a pre-trained language model to generate the soft prompt and optimizes it
# # using cosine similarity with the direction vector.

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import pickle
# import os

# # ---------------------------- CONFIG ----------------------------
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# PKL_PATH = "directions/rfm_harmful_llama_3_8b_it.pkl"  # path to your direction-only pkl
# TARGET_LAYER = -1  # layer to align with (e.g., -1 for last hidden layer)
# N_TOKENS = 10  # number of soft prompt tokens
# LR = 5e-3 #5e-2
# STEPS = 500
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ---------------------------- LOAD MODEL ----------------------------
# print("Loading model and tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     torch_dtype=torch.float32,
# )
# model.eval()

# embedding_layer = model.get_input_embeddings()  # nn.Embedding
# hidden_size = embedding_layer.embedding_dim

# # ---------------------------- LOAD DIRECTION ----------------------------
# print("Loading direction from pkl file...")
# with open(PKL_PATH, 'rb') as f:
#     directions = pickle.load(f)

# if TARGET_LAYER not in directions:
#     raise ValueError(f"Layer {TARGET_LAYER} not found in .pkl file")

# # direction_vec = directions[TARGET_LAYER][0].to(DEVICE)  # use first component
# direction_vec = directions[TARGET_LAYER][0].to(DEVICE).to(torch.float32)


# # ---------------------------- INIT SOFT PROMPT ----------------------------
# print("Initializing soft prompt...")
# # soft_prompt = nn.Parameter(torch.randn(N_TOKENS, hidden_size, device=DEVICE) * 0.01)
# soft_prompt = nn.Parameter(
#     torch.randn(N_TOKENS, hidden_size, device=DEVICE, dtype=torch.float32) * 0.01
# )
# optimizer = torch.optim.Adam([soft_prompt], lr=LR)

# # ---------------------------- FORWARD PASS ----------------------------
# def forward_pass(soft_prompt):
#     input_ids = tokenizer("The", return_tensors='pt').input_ids.to(DEVICE)
#     embedded_input = embedding_layer(input_ids)  # (1, seq_len, hidden_size)

#     full_input = torch.cat([soft_prompt.unsqueeze(0), embedded_input], dim=1)  # (1, N+T, H)
#     outputs = model(inputs_embeds=full_input, output_hidden_states=True)
#     hidden_states = outputs.hidden_states[TARGET_LAYER]  # (1, seq_len, hidden_size)

#     return hidden_states[:, 0, :]  # first soft token

# # ---------------------------- OPTIMIZATION LOOP ----------------------------
# print("Starting optimization...")
# for step in range(STEPS):
#     optimizer.zero_grad()
#     activation = forward_pass(soft_prompt)  # (1, hidden_size)

#     # cosine_loss = -F.cosine_similarity(activation, direction_vec.unsqueeze(0)).mean()
#     cos_sim = F.cosine_similarity(activation, direction_vec.unsqueeze(0)).clamp(-1.0, 1.0)
#     cosine_loss = -cos_sim.mean()
#     cosine_loss.backward()
#     optimizer.step()

#     if step % 50 == 0 or step == STEPS - 1:
#         print(f"Step {step}: Cosine loss = {cosine_loss.item():.5f}")

# # ---------------------------- DECODE PROMPT ----------------------------
# print("Decoding prompt from optimized soft embeddings...")
# with torch.no_grad():
#     embedding_weights = embedding_layer.weight  # (vocab_size, hidden_size)
#     decoded_ids = []

#     for token_emb in soft_prompt:
#         similarities = F.cosine_similarity(token_emb.unsqueeze(0), embedding_weights)
#         token_id = similarities.argmax().item()
#         decoded_ids.append(token_id)

#     decoded_prompt = tokenizer.decode(decoded_ids)
#     print("\n\u2728 Final Generated Prompt:")
#     print(decoded_prompt)

# # Optional: Save prompt embedding
# os.makedirs("optimized_prompts", exist_ok=True)
# torch.save(soft_prompt.detach().cpu(), f"optimized_prompts/prompt_layer{TARGET_LAYER}.pt")

# #### # decode_pkl.py
# # This script generates a soft prompt aligned with a specified direction vector from a .pkl file
# # using a pre-trained language model. It optimizes the prompt to maximize cosine similarity with the
# # direction vector and decodes the resulting embeddings into a human-readable prompt.

# # Best result: Score: 0.0466 | Prompt: A dangerous action someone might take: I'm worried that if they start to drive while they're intoxicated, they might get into an accident and hurt themselves or someone else.
# # A sudden and often violent action: The crowd's frustration boiled over into a riot when the game was called off due to a disputed call.
# # A violent or aggressive action: The angry customer lashed out at the store clerk, shouting insults and making a scene.
# # A sudden and often violent action: The earthquake triggered a landslide that swept through the village, destroying homes and infrastructure.
# # A sudden and often violent action: The

# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm
# import pickle
# import random
# import os

# # ---------------------------- CONFIG ----------------------------
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# PKL_PATH = "directions/rfm_harmful_llama_3_8b_it.pkl"
# TARGET_LAYER = -1
# DIRECTION_IDX = 0
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TOP_K = 10
# N_DYNAMIC_SEEDS = 300  # Increased for better coverage
# COMPLETIONS_PER_SEED = 50  # More completions per seed
# MAX_NEW_TOKENS = 80  # Allow longer completions

# # ---------------------------- LOAD MODEL ----------------------------
# print("Loading model and tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     torch_dtype=torch.float16,
# )
# model.eval()

# # ---------------------------- LOAD DIRECTION ----------------------------
# print("Loading direction from pkl file...")
# with open(PKL_PATH, 'rb') as f:
#     directions = pickle.load(f)

# direction_vec = directions[TARGET_LAYER][DIRECTION_IDX].to(DEVICE).to(torch.float32)
# direction_vec = F.normalize(direction_vec, dim=0)

# # ---------------------------- DYNAMIC SEEDING ----------------------------
# print("Generating high-quality dynamic seeds...")
# base_templates = [
#     "The following statement expresses a harmful idea:",
#     "An unethical recommendation:",
#     "A dangerous action someone might take:",
#     "A sentence likely to result in negative outcomes:",
#     "An unsafe behavior:",
#     "This advice should not be followed:",
#     "A scenario that causes problems:",
#     "The worst possible suggestion:",
#     "Harmful instructions:",
#     "A malicious command:"
# ]

# # Expand dynamic seed prompts using LLaMA itself
# seed_prompts = []
# for template in base_templates:
#     input_ids = tokenizer(template, return_tensors='pt').input_ids.to(DEVICE)
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=input_ids,
#             max_new_tokens=30,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.95,
#             num_return_sequences=N_DYNAMIC_SEEDS // len(base_templates),
#             pad_token_id=tokenizer.eos_token_id
#         )
#         for out in outputs:
#             decoded = tokenizer.decode(out, skip_special_tokens=True).strip()
#             if len(decoded.split()) > 6:
#                 seed_prompts.append(decoded)

# print(f"Generated {len(seed_prompts)} refined seeds.")

# # ---------------------------- EXPAND SEEDS INTO COMPLETIONS ----------------------------
# print("Generating completions from seed prompts...")
# candidate_prompts = []

# for seed in tqdm(seed_prompts):
#     input_ids = tokenizer(seed, return_tensors='pt').input_ids.to(DEVICE)
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=input_ids,
#             max_new_tokens=MAX_NEW_TOKENS,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.95,
#             num_return_sequences=COMPLETIONS_PER_SEED,
#             pad_token_id=tokenizer.eos_token_id
#         )
#         for output in outputs:
#             prompt = tokenizer.decode(output, skip_special_tokens=True).strip()
#             if len(prompt.split()) > 8 and not prompt.endswith(":"):
#                 candidate_prompts.append(prompt)

# print(f"Total candidate prompts: {len(candidate_prompts)}")

# # ---------------------------- COSINE EVALUATION ----------------------------
# def get_hidden_direction(prompt):
#     inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#         hidden = outputs.hidden_states[TARGET_LAYER]  # (1, seq_len, hidden_size)
#         rep = hidden[0][-1].to(torch.float32)  # use last token instead of mean
#         return rep

# print(f"Evaluating prompts against direction vector...")
# prompt_scores = []

# for prompt in tqdm(candidate_prompts):
#     try:
#         hidden = get_hidden_direction(prompt)
#         cos_sim = F.cosine_similarity(hidden, direction_vec, dim=0).item()
#         prompt_scores.append((cos_sim, prompt))
#     except Exception:
#         continue

# # ---------------------------- RESULTS ----------------------------
# top_prompts = sorted(prompt_scores, key=lambda x: -x[0])[:TOP_K]

# print("\n✨ Top Matching Prompts:\n")
# for score, prompt in top_prompts:
#     print(f"Score: {score:.4f} | Prompt: {prompt}")

# # Optional: Save results
# os.makedirs("best_prompts", exist_ok=True)
# with open("best_prompts/top_prompts.txt", "w") as f:
#     for score, prompt in top_prompts:
#         f.write(f"{score:.4f}\t{prompt}\n")

##### CLAUDE #########
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pickle
import random
import os
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------------------- CONFIG ----------------------------
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PKL_PATH = "directions/rfm_harmful_llama_3_8b_it.pkl"
TARGET_LAYER = -1
DIRECTION_IDX = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 10

# ---------------------------- LOAD MODEL ----------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float32,
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------- LOAD DIRECTION ----------------------------
print("Loading direction from pkl file...")
with open(PKL_PATH, 'rb') as f:
    directions = pickle.load(f)

direction_vec = directions[TARGET_LAYER][DIRECTION_IDX].to(DEVICE).to(torch.float32)
print(f"Direction vector shape: {direction_vec.shape}")
print(f"Direction vector norm: {torch.norm(direction_vec):.4f}")

# ---------------------------- DIRECT EMBEDDING OPTIMIZATION ----------------------------
def optimize_embedding_directly(direction_vec, vocab_size=32000, sequence_length=20, num_iterations=1000):
    """Directly optimize token embeddings to match the direction vector"""
    print("\n=== DIRECT EMBEDDING OPTIMIZATION ===")
    
    # Get embedding matrix
    embedding_matrix = model.get_input_embeddings().weight.data.clone()
    
    # Find the single token that's most similar to the direction
    direction_norm = F.normalize(direction_vec.unsqueeze(0), dim=1)
    embeddings_norm = F.normalize(embedding_matrix, dim=1)
    similarities = torch.mm(direction_norm, embeddings_norm.t()).squeeze()
    
    # Get top tokens
    top_k = min(1000, vocab_size)
    top_indices = similarities.topk(top_k).indices
    top_similarities = similarities.topk(top_k).values
    
    print(f"Top single token similarity: {top_similarities[0]:.4f}")
    
    # Try different combinations of top tokens
    best_combinations = []
    
    for seq_len in [1, 2, 3, 4, 5, 8, 10, 15, 20]:
        print(f"Testing sequences of length {seq_len}...")
        
        # Try random combinations of top tokens
        for _ in range(min(1000, 10000 // seq_len)):
            # Sample tokens from top candidates (weighted by similarity)
            probs = F.softmax(top_similarities * 10, dim=0)  # Sharpen distribution
            sampled_indices = torch.multinomial(probs, seq_len, replacement=True)
            token_ids = top_indices[sampled_indices]
            
            # Convert to text and get representation
            try:
                tokens = [tokenizer.decode([tid.item()]) for tid in token_ids]
                prompt = "".join(tokens)
                
                # Get hidden representation
                inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[TARGET_LAYER]
                    
                    # Try different aggregation methods
                    representations = {
                        'last': hidden[0, -1],
                        'mean': hidden[0].mean(dim=0),
                        'max': hidden[0].max(dim=0)[0],
                        'first': hidden[0, 0]
                    }
                    
                    for method, rep in representations.items():
                        cos_sim = F.cosine_similarity(rep.to(torch.float32), direction_vec, dim=0).item()
                        best_combinations.append((cos_sim, prompt, method, seq_len))
                        
            except Exception as e:
                continue
    
    return sorted(best_combinations, key=lambda x: -x[0])

# ---------------------------- ACTIVATION MAXIMIZATION ----------------------------
def activation_maximization(direction_vec, learning_rate=0.001, num_iterations=2000):
    """Use activation maximization to find optimal input"""
    print("\n=== ACTIVATION MAXIMIZATION ===")
    
    # Start with learnable token embeddings
    vocab_size = tokenizer.vocab_size
    sequence_length = 15
    
    # Initialize with tokens most similar to direction
    embedding_matrix = model.get_input_embeddings().weight.data
    direction_norm = F.normalize(direction_vec.unsqueeze(0), dim=1)
    embeddings_norm = F.normalize(embedding_matrix, dim=1)
    similarities = torch.mm(direction_norm, embeddings_norm.t()).squeeze()
    
    # Start with top tokens
    top_indices = similarities.topk(sequence_length).indices
    initial_embeddings = embedding_matrix[top_indices].unsqueeze(0)  # (1, seq_len, hidden_size)
    
    # Make embeddings learnable
    optimized_embeddings = initial_embeddings.clone().detach().requires_grad_(True)
    
    # Optimizer
    optimizer = torch.optim.Adam([optimized_embeddings], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
    
    best_score = -float('inf')
    best_embeddings = None
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs_embeds=optimized_embeddings, output_hidden_states=True)
        hidden = outputs.hidden_states[TARGET_LAYER]
        
        # Get representation (try multiple methods)
        rep_last = hidden[0, -1].to(torch.float32)
        rep_mean = hidden[0].mean(dim=0).to(torch.float32)
        
        # Calculate similarities
        cos_sim_last = F.cosine_similarity(rep_last, direction_vec, dim=0)
        cos_sim_mean = F.cosine_similarity(rep_mean, direction_vec, dim=0)
        
        # Objective: maximize cosine similarity
        objective = cos_sim_last + cos_sim_mean
        loss = -objective
        
        # Add regularization to keep embeddings reasonable
        embedding_norm_penalty = 0.001 * torch.norm(optimized_embeddings)
        loss += embedding_norm_penalty
        
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_([optimized_embeddings], max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        current_score = objective.item()
        if current_score > best_score:
            best_score = current_score
            best_embeddings = optimized_embeddings.clone().detach()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Score = {current_score:.4f}, Best = {best_score:.4f}")
    
    # Convert best embeddings back to tokens
    if best_embeddings is not None:
        with torch.no_grad():
            # Find closest tokens for each position
            best_tokens = []
            for i in range(sequence_length):
                emb = best_embeddings[0, i]
                similarities = torch.mm(emb.unsqueeze(0), embedding_matrix.t())
                best_token_id = similarities.argmax().item()
                token = tokenizer.decode([best_token_id])
                best_tokens.append(token)
            
            optimized_prompt = "".join(best_tokens)
            
            # Verify score
            inputs = tokenizer(optimized_prompt, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[TARGET_LAYER]
            
            rep_last = hidden[0, -1].to(torch.float32)
            rep_mean = hidden[0].mean(dim=0).to(torch.float32)
            
            final_score_last = F.cosine_similarity(rep_last, direction_vec, dim=0).item()
            final_score_mean = F.cosine_similarity(rep_mean, direction_vec, dim=0).item()
            
            return [(final_score_last, optimized_prompt, 'last'), 
                   (final_score_mean, optimized_prompt, 'mean')]
    
    return []

# ---------------------------- GENETIC ALGORITHM ----------------------------
def genetic_algorithm(direction_vec, population_size=100, generations=50, mutation_rate=0.1):
    """Use genetic algorithm to evolve optimal token sequences"""
    print("\n=== GENETIC ALGORITHM ===")
    
    # Get top tokens for initialization
    embedding_matrix = model.get_input_embeddings().weight.data
    direction_norm = F.normalize(direction_vec.unsqueeze(0), dim=1)
    embeddings_norm = F.normalize(embedding_matrix, dim=1)
    similarities = torch.mm(direction_norm, embeddings_norm.t()).squeeze()
    
    top_k = 500
    top_indices = similarities.topk(top_k).indices.cpu().numpy()
    
    # Initialize population
    sequence_length = 12
    population = []
    
    for _ in range(population_size):
        # Create random sequence from top tokens
        sequence = np.random.choice(top_indices, sequence_length)
        population.append(sequence)
    
    def evaluate_fitness(sequence):
        """Evaluate fitness of a token sequence"""
        try:
            tokens = [tokenizer.decode([int(tid)]) for tid in sequence]
            prompt = "".join(tokens)
            
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[TARGET_LAYER]
                
                rep_last = hidden[0, -1].to(torch.float32)
                cos_sim = F.cosine_similarity(rep_last, direction_vec, dim=0).item()
                
                return cos_sim
        except:
            return -1.0
    
    def crossover(parent1, parent2):
        """Create offspring from two parents"""
        crossover_point = random.randint(1, sequence_length - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def mutate(sequence):
        """Mutate a sequence"""
        mutated = sequence.copy()
        for i in range(sequence_length):
            if random.random() < mutation_rate:
                mutated[i] = random.choice(top_indices)
        return mutated
    
    best_fitness = -float('inf')
    best_sequence = None
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_fitness(seq) for seq in population]
        
        # Track best
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_sequence = population[fitness_scores.index(max_fitness)]
        
        print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        # Selection (tournament selection)
        new_population = []
        
        for _ in range(population_size // 2):
            # Select parents
            parent1 = population[random.choices(range(population_size), weights=[max(0, f) for f in fitness_scores])[0]]
            parent2 = population[random.choices(range(population_size), weights=[max(0, f) for f in fitness_scores])[0]]
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.extend([child1, child2])
        
        population = new_population
    
    # Return best result
    if best_sequence is not None:
        tokens = [tokenizer.decode([int(tid)]) for tid in best_sequence]
        prompt = "".join(tokens)
        return [(best_fitness, prompt, 'genetic')]
    
    return []

# ---------------------------- GRADIENT-FREE OPTIMIZATION ----------------------------
def gradient_free_optimization(direction_vec, num_iterations=5000):
    """Use gradient-free optimization methods"""
    print("\n=== GRADIENT-FREE OPTIMIZATION ===")
    
    # Get top tokens
    embedding_matrix = model.get_input_embeddings().weight.data
    direction_norm = F.normalize(direction_vec.unsqueeze(0), dim=1)
    embeddings_norm = F.normalize(embedding_matrix, dim=1)
    similarities = torch.mm(direction_norm, embeddings_norm.t()).squeeze()
    
    top_k = 1000
    top_indices = similarities.topk(top_k).indices.cpu().numpy()
    top_weights = F.softmax(similarities.topk(top_k).values * 5, dim=0).cpu().numpy()
    
    def objective_function(token_sequence):
        """Objective function to maximize"""
        try:
            tokens = [tokenizer.decode([int(tid)]) for tid in token_sequence]
            prompt = "".join(tokens)
            
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[TARGET_LAYER]
                
                # Try multiple aggregation methods
                scores = []
                for method in ['last', 'mean', 'max']:
                    if method == 'last':
                        rep = hidden[0, -1]
                    elif method == 'mean':
                        rep = hidden[0].mean(dim=0)
                    elif method == 'max':
                        rep = hidden[0].max(dim=0)[0]
                    
                    cos_sim = F.cosine_similarity(rep.to(torch.float32), direction_vec, dim=0).item()
                    scores.append(cos_sim)
                
                return max(scores)
        except:
            return -1.0
    
    # Simulated annealing
    sequence_length = 10
    current_sequence = np.random.choice(top_indices, sequence_length, p=top_weights)
    current_score = objective_function(current_sequence)
    
    best_sequence = current_sequence.copy()
    best_score = current_score
    
    temperature = 1.0
    cooling_rate = 0.995
    
    for iteration in range(num_iterations):
        # Generate neighbor
        neighbor = current_sequence.copy()
        
        # Random modification
        if random.random() < 0.5:
            # Replace random token
            pos = random.randint(0, sequence_length - 1)
            neighbor[pos] = np.random.choice(top_indices, p=top_weights)
        else:
            # Swap two positions
            pos1, pos2 = random.sample(range(sequence_length), 2)
            neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
        
        neighbor_score = objective_function(neighbor)
        
        # Accept or reject
        if neighbor_score > current_score or random.random() < np.exp((neighbor_score - current_score) / temperature):
            current_sequence = neighbor
            current_score = neighbor_score
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_sequence = current_sequence.copy()
        
        # Cool down
        temperature *= cooling_rate
        
        if iteration % 500 == 0:
            print(f"Iteration {iteration}: Current = {current_score:.4f}, Best = {best_score:.4f}")
    
    # Convert best sequence to prompt
    tokens = [tokenizer.decode([int(tid)]) for tid in best_sequence]
    prompt = "".join(tokens)
    
    return [(best_score, prompt, 'simulated_annealing')]

# ---------------------------- MAIN EXECUTION ----------------------------
def main():
    print("\n" + "="*60)
    print("ULTRA-AGGRESSIVE OPTIMIZATION FOR SCORE ~1.0")
    print("="*60)
    
    all_results = []
    
    # Phase 1: Direct embedding optimization
    print("\n🚀 Phase 1: Direct Embedding Optimization")
    direct_results = optimize_embedding_directly(direction_vec)
    all_results.extend(direct_results[:20])
    if direct_results:
        print(f"Best direct score: {direct_results[0][0]:.4f}")
    
    # Phase 2: Activation maximization
    print("\n🚀 Phase 2: Activation Maximization")
    activation_results = activation_maximization(direction_vec)
    all_results.extend(activation_results)
    if activation_results:
        print(f"Best activation score: {max(r[0] for r in activation_results):.4f}")
    
    # Phase 3: Genetic algorithm
    print("\n🚀 Phase 3: Genetic Algorithm")
    genetic_results = genetic_algorithm(direction_vec)
    all_results.extend(genetic_results)
    if genetic_results:
        print(f"Best genetic score: {genetic_results[0][0]:.4f}")
    
    # Phase 4: Gradient-free optimization
    print("\n🚀 Phase 4: Gradient-Free Optimization")
    gradient_free_results = gradient_free_optimization(direction_vec)
    all_results.extend(gradient_free_results)
    if gradient_free_results:
        print(f"Best gradient-free score: {gradient_free_results[0][0]:.4f}")
    
    # Final ranking
    final_results = sorted(all_results, key=lambda x: -x[0])[:TOP_K]
    
    print("\n" + "="*60)
    print("🏆 FINAL ULTRA-OPTIMIZED RESULTS:")
    print("="*60)
    
    for i, (score, prompt, method, *extra) in enumerate(final_results):
        print(f"{i+1:2d}. Score: {score:.6f} | Method: {method} | Prompt: {repr(prompt)}")
    
    # Save results
    os.makedirs("ultra_results", exist_ok=True)
    with open("ultra_results/ultra_optimized.txt", "w") as f:
        for result in final_results:
            f.write(f"{result[0]:.6f}\t{result[1]}\t{result[2]}\n")
    
    if final_results:
        best_score = final_results[0][0]
        print(f"\n🎯 BEST ACHIEVED SCORE: {best_score:.6f}")
        
        if best_score > 0.95:
            print("🏆 PERFECT MATCH ACHIEVED!")
        elif best_score > 0.8:
            print("🌟 EXCELLENT MATCH!")
        elif best_score > 0.6:
            print("✅ VERY GOOD MATCH!")
        elif best_score > 0.4:
            print("📈 GOOD IMPROVEMENT!")
        else:
            print("🔧 Need more aggressive optimization")
    
    return final_results

if __name__ == "__main__":
    main()
