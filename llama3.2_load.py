from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_file_1 = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="model-00001-of-00002.safetensors"
)
model_file_2 = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="model-00002-of-00002.safetensors"
)
index_file = hf_hub_download(
    repo_id="meta-llama/Llama-3.2-3B", 
    filename="model.safetensors.index.json"
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")