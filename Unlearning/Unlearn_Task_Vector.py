import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Configuration Dataclass (As provided by user) ---
@dataclass
class ModelConfig:
    name: str
    model_path: str
    is_local: bool = True
    model_type: Literal["causal", "encoder", "encoder-decoder"] = "causal" # Keep for potential future use
    is_adapter_model: bool = False
    base_model_path_for_adapter: Optional[str] = None

    def __post_init__(self):
        if self.is_adapter_model and not self.base_model_path_for_adapter:
            raise ValueError(f"Model '{self.name}' is marked as adapter model, but 'base_model_path_for_adapter' is not provided.")
        if self.is_adapter_model and self.base_model_path_for_adapter == self.model_path:
             raise ValueError(f"For adapter model '{self.name}', 'base_model_path_for_adapter' cannot be the same as 'model_path'.")

MODEL_CONFIGS = [
    # Add the models you want to process here
    # ModelConfig(
    #     name="Llamas-3.2-3B_ENG", # Name of the *fine-tuned* model to start unlearning from
    #     model_path="/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Llamas-3.2-3B/Full_TOFU_Llamas-3.2-3B_ENG", # Path to the fine-tuned model
    #     is_local=True,
    #     is_adapter_model=False, # Assuming this is a fully fine-tuned model, not just adapters
    # ),
    # ModelConfig(
    #     name="Llamas-3.2-3B_ALL", # Name of the *fine-tuned* model to start unlearning from
    #     model_path="/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Llamas-3.2-3B/Full_TOFU_Llamas-3.2-3B_ALL", # Path to the fine-tuned model
    #     is_local=True,
    #     is_adapter_model=False, # Assuming this is a fully fine-tuned model, not just adapters
    # ),
    ModelConfig(
        name="Qwen2.5-7B-Instruct_ENG",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Qwen2.5-7B-Instruct/Full_TOFU_Qwen2.5-7B-Instruct_ENG",
        is_local=True,
        is_adapter_model=False,
    ),
    ModelConfig(
        name="Qwen2.5-7B-Instruct_ALL",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Qwen2.5-7B-Instruct/Full_TOFU_Qwen2.5-7B-Instruct_ALL",
        is_local=True,
        is_adapter_model=False,
    ),
    ModelConfig(
        name="gemma-3-4B-it_ENG",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Gemma-3-4B-it/Full_TOFU_Gemma-3-4B-it_ENG",
        is_local=True,
        is_adapter_model=False,
    ),
    ModelConfig(
        name="gemma-3-4B-it_ALL",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Gemma-3-4B-it/Full_TOFU_Gemma-3-4B-it_ALL",
        is_local=True,
        is_adapter_model=False,
    ),
]

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

# --- Main Execution Logic ---
def main():
    # Task Vector specific hyperparameter
    ALPHA = -3.0 # As used in the original example, adjust as needed
    # Note: The interpretation of alpha depends on how task vector is defined.
    # If TV = Ref - Pre, then Unlearned = Pre + alpha * (Ref - Pre).
    # alpha = +1 -> Unlearned = Ref
    # alpha = 0 -> Unlearned = Pre
    # alpha = -1 -> Unlearned = 2*Pre - Ref (extrapolates away from Ref)

    # Base directory for saving unlearned models
    base_output_dir = "./unlearned_models_taskvector"

    logger.info(f"Starting Task Vector Unlearning process for {len(MODEL_CONFIGS)} models...")
    overall_start_time = time.time()

    for config in MODEL_CONFIGS:
        model_start_time = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing Model: {config.name}")
        logger.info(f"  Target Model Path (Pretrained for TV): {config.model_path}")
        logger.info(f"  Reference Model Path (Finetuned for TV): {config.reference_model_path}")
        logger.info(f"{'='*50}")

        # Define unique output path for this model and alpha
        output_model_path = os.path.join(base_output_dir, f"unlearned_{config.name}_alpha{ALPHA}")

        unlearner = None # Ensure unlearner is reset or None initially

        try:
            # Check if reference path exists (basic check)
            if config.is_local and not os.path.exists(config.reference_model_path):
                logger.error(f"Reference model path not found: {config.reference_model_path}. Skipping {config.name}.")
                continue
            if config.is_local and not os.path.exists(config.model_path):
                 logger.error(f"Target model path not found: {config.model_path}. Skipping {config.name}.")
                 continue

            # Initialize unlearner
            unlearner = TaskVectorUnlearner(
                pretrained_model_path=config.model_path,    # Passed as 'pretrained' to the class
                finetuned_model_path=config.reference_model_path # Passed as 'finetuned' to the class
            )

            # Apply unlearning and save model
            unlearner.apply_unlearning(alpha=ALPHA)
            unlearner.save_unlearned_model(output_model_path)

            # --- Optional: Generate answers using the unlearned model ---
            # if you have forget/retain datasets loaded:
            logger.info(f"Generating answers for {config.name} (if datasets available)...")
            forget_answers_file = os.path.join(output_model_path, "forget_answers_generated.json")
            retain_answers_file = os.path.join(output_model_path, "retain_answers_generated.json")
            if 'forget_dataset' in locals():
                generate_and_save_answers(unlearner, forget_dataset, forget_answers_file)
            if 'retain_sample' in locals():
                generate_and_save_answers(unlearner, retain_sample, retain_answers_file)
            else:
                logger.warning("Forget/Retain datasets not loaded, skipping answer generation.")
            # --------------------------------------------------------------

            logger.info(f"Successfully processed and saved model: {config.name}")

        except Exception as e:
            logger.error(f"CRITICAL ERROR during processing model {config.name}: {e}")
            logger.error(traceback.format_exc())
            # Optionally save error state for this model
            error_dir = os.path.join(base_output_dir, "errors")
            os.makedirs(error_dir, exist_ok=True)
            error_file_path = os.path.join(error_dir, f"ERROR_{config.name}.log")
            try:
                 with open(error_file_path, "w") as ef:
                     ef.write(f"Error processing model: {config.name}\n")
                     ef.write(f"Config:\n  Model Path: {config.model_path}\n  Reference Path: {config.reference_model_path}\n")
                     ef.write(f"Alpha: {ALPHA}\n")
                     ef.write(f"Error: {e}\n")
                     ef.write(traceback.format_exc())
                 logger.info(f"Error log saved to {error_file_path}")
            except Exception as log_e:
                 logger.error(f"Failed to write error log: {log_e}")

        finally:
            # --- Memory Cleanup after each model ---
            logger.info(f"Cleaning up resources for model {config.name}...")
            if unlearner is not None:
                del unlearner.pretrained_model
                del unlearner.finetuned_model
                if unlearner.unlearned_model is not None:
                    del unlearner.unlearned_model
                del unlearner
                logger.debug("Deleted unlearner object and its models.")

            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared.")
                except Exception as cuda_e:
                    logger.warning(f"Could not clear CUDA cache: {cuda_e}")
            logger.info(f"Finished cleanup for {config.name}.")
            model_end_time = time.time()
            logger.info(f"Time taken for model {config.name}: {model_end_time - model_start_time:.2f} seconds.")
            time.sleep(2) # Small delay before next model


    overall_end_time = time.time()
    logger.info(f"\n{'='*50}")
    logger.info(f"Finished processing all models.")
    logger.info(f"Total time: {overall_end_time - overall_start_time:.2f} seconds.")
    logger.info(f"Unlearned models saved in: {base_output_dir}")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    main()