import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

class TaskVectorUnlearner:
    def __init__(
        self,
        pretrained_model_path,
        finetuned_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load models and tokenizers
        print("Loading pretrained model (100% TOFU)...")
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        
        print("Loading finetuned model (99% TOFU)...")
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            finetuned_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.unlearned_model = None
    
    def compute_task_vector(self):
        print("Computing task vector...")
        task_vector = {}
        
        # Get model parameters
        pretrained_params = dict(self.pretrained_model.named_parameters())
        
        # Compute task vector: finetuned - pretrained
        for name, param in tqdm(self.finetuned_model.named_parameters(), desc="Computing task vector"):
            if name in pretrained_params:
                task_vector[name] = param.data - pretrained_params[name].data
        
        return task_vector
    
    def apply_unlearning(self, alpha):
        print(f"Applying unlearning with alpha={alpha}...")
        
        task_vector = self.compute_task_vector()
        
        # Create a copy of the pretrained model
        self.unlearned_model = type(self.pretrained_model).from_pretrained(
            self.pretrained_model.config._name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        # Apply task vector negation: pretrained - alpha * task_vector
        unlearned_dict = self.unlearned_model.state_dict()
        pretrained_dict = self.pretrained_model.state_dict()
        
        with torch.no_grad():
            for name, param in tqdm(self.unlearned_model.named_parameters(), desc="Applying unlearning"):
                if name in task_vector:
                    unlearned_dict[name] = pretrained_dict[name] + alpha * task_vector[name]
        
        self.unlearned_model.load_state_dict(unlearned_dict)
        print("Unlearning complete!")
    
    def save_unlearned_model(self, output_path):
        if self.unlearned_model is None:
            raise ValueError("Unlearned model not created yet. Call apply_unlearning first.")
        
        print(f"Saving unlearned model to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        self.unlearned_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Model saved successfully!")
    
    def generate_answer(self, question, max_new_tokens=256):
        if self.unlearned_model is None:
            raise ValueError("Unlearned model not created yet. Call apply_unlearning first.")
        
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.unlearned_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode response (excluding the prompt)
        generated_text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text

def load_tofu_dataset(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def split_dataset(dataset, forget_ratio=0.01, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Make a copy to avoid modifying the original
    dataset_copy = dataset.copy()
    
    # Determine number of examples to forget
    n_examples = len(dataset_copy)
    n_forget = int(n_examples * forget_ratio)
    
    # Randomly select indices to forget
    all_indices = list(range(n_examples))
    forget_indices = set(np.random.choice(all_indices, size=n_forget, replace=False))
    
    # Create the split datasets
    forget_dataset = [dataset_copy[i] for i in range(n_examples) if i in forget_indices]
    retain_dataset = [dataset_copy[i] for i in range(n_examples) if i not in forget_indices]
    
    print(f"Dataset split: {len(forget_dataset)} examples to forget, {len(retain_dataset)} examples to retain")
    return forget_dataset, retain_dataset

def generate_and_save_answers(unlearner, dataset, output_file):
    results = []
    
    for item in tqdm(dataset, desc="Generating answers"):
        question = item["question"]
        ground_truth = item["answer"]
        
        # Generate answer using unlearned model
        generated_answer = unlearner.generate_answer(question)
        
        # Store results
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer
        })
    
    # Save results to JSON file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Generated answers saved to {output_file}")
    return results

def main():
    # Paths - update these with your actual paths
    # tofu_data_path = "/scratch/nchaud28/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train/full.json"  # Update with your local path
    pretrained_model_path = "/data/courses/2025/class_cse576spring2025_vgupt140/Axolotls/FineTuning/TOFU_Llamas-3.2-3B/Full_TOFU_Llamas-3.2-3B_ENG"  # Update with your model path
    finetuned_model_path = "/data/courses/2025/class_cse576spring2025_vgupt140/Axolotls/FineTuning/TOFU_Llamas-3.2-3B/Full_TOFU_Llamas-3.2-3B_Retain99"  # Update with your model path
    output_model_path = "/scratch/nchaud28/De-fine-tuning-Unlearning-Multilingual-Language-Models/Unlearning_Models/llama3.2_ENG_3.0"
    
    # Output files for generated answers
    # forget_answers_file = "/scratch/nchaud28/De-fine-tuning-Unlearning-Multilingual-Language-Models/Unlearning_Models/outputs/llama/forget_answers.json"
    # retain_answers_file = "/scratch/nchaud28/De-fine-tuning-Unlearning-Multilingual-Language-Models/Unlearning_Models/outputs/llama/retain_answers.json"
    
    # Load TOFU dataset
    # print(f"Loading TOFU dataset from {tofu_data_path}...")
    # tofu_dataset = load_tofu_dataset(tofu_data_path)
    
    # # Split dataset (1% to forget, 99% to retain)
    # forget_dataset, retain_dataset = split_dataset(tofu_dataset, forget_ratio=0.01, seed=42)
    
    # Initialize unlearner
    unlearner = TaskVectorUnlearner(
        pretrained_model_path=pretrained_model_path,
        finetuned_model_path=finetuned_model_path
    )
    
    # Apply unlearning and save model
    unlearner.apply_unlearning(alpha=-3.0)  # Negative alpha for unlearning
    unlearner.save_unlearned_model(output_model_path)
    
    # Generate and save answers
    # print("Generating answers for 'forget' dataset...")
    # generate_and_save_answers(unlearner, forget_dataset, forget_answers_file)
    
    # print("Generating answers for sample of 'retain' dataset...")
    # # Use a smaller sample of the retain dataset to save time (adjust as needed)
    # retain_sample = retain_dataset[:100]  # Just use 100 examples from retain dataset
    # generate_and_save_answers(unlearner, retain_sample, retain_answers_file)
    
    print("Done!")

if __name__ == "__main__":
    main()