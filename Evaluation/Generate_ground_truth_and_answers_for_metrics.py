import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import os

# --- Configuration ---
MODEL_PATH = "/data/courses/2025/class_cse576spring2025_vgupt140/Axolotls/KL_unlearning_Gemma/KD_forget01_unlearned_model"
INPUT_DIR = "/scratch/smohan63/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train/"  # Directory containing JSON files
OUTPUT_DIR = "/scratch/smohan63/De-fine-tuning-Unlearning-Multilingual-Language-Models/Evaluation/TOFU_Evaluation_Results_epoch3_bert_Gemma"  # Directory to save output JSONs
MAX_NEW_TOKENS = 150
BATCH_SIZE = 16

# --- Load Model and Tokenizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Process Each JSON File in Directory ---
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".json"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle format: list of dicts with "question" and "answer"
    if isinstance(data, list):
        data = [item for item in data if isinstance(item, dict) and "question" in item and "answer" in item]
    else:
        print(f"Skipping file {filename} due to unexpected format.")
        continue

    questions = [item["question"] for item in data]
    ground_truths = [item["answer"] for item in data]

    generated_answers = []
    num_batches = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(num_batches), desc=f"Processing {filename}"):
        batch_questions = questions[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        prompts = [f"Question: {q}\nAnswer:" for q in batch_questions]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[1]
        for i in range(outputs.shape[0]):
            generated_ids = outputs[i, input_length:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            print(f"Generated answer: {answer}")
            generated_answers.append(answer)

    output_data = [
        {
            "question": q,
            "ground_truth": gt,
            "generated_answer": ga
        }
        for q, gt, ga in zip(questions, ground_truths, generated_answers)
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved generated outputs to {output_path}")