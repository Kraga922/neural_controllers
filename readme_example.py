from neural_controllers import NeuralController
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import os

# Optional: helps avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize tokenizer and model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

language_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",                # spread across available GPUs
    torch_dtype=torch.float16,        # cut memory usage
    low_cpu_mem_usage=True,           # reduces RAM usage while loading
    offload_folder="offload"          # spill large parts to disk if needed
)

# Create neural controller
controller = NeuralController(
    language_model,
    tokenizer,
    rfm_iters=8,
    batch_size=2,
    n_components=5,
    control_method='rfm'
)

# ...existing code...

if not os.path.exists('../directions/rfm_shakespeare_llama_3_8b_it.pkl'):
    from utils import shakespeare_dataset
    data_dir = "../data/shakespeare"
    assistant_tag = "assistant"  # or the correct tag for your dataset
    dataset = shakespeare_dataset(data_dir, tokenizer, controller, assistant_tag)
    train_data = dataset['shakespeare']['train']
    controller.compute_directions(train_data['inputs'], train_data['labels'])
    controller.save(concept='shakespeare', model_name='llama_3_8b_it', path='../directions/')

# ...existing code...

# Load pre-trained directions
controller.load(concept='shakespeare', model_name='llama_3_8b_it', path='../directions/')

# Generate controlled text
prompt = controller.format_prompt("What can I do to treat flu symptoms?")
controlled_output = controller.generate(
    prompt,
    layers_to_control=list(range(-1, -31, -1)),
    control_coef=0.5,
    max_new_tokens=150
)
