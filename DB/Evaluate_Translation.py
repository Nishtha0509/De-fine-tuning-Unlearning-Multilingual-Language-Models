from bert_score import score
import json
import os

# file load
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# file paths
en_path = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full.json"
ko_path = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full_korean_mistral.json"
hi_path = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full_hindi_mistral.json"

# data load
en_data = load_json(en_path)
ko_data = load_json(ko_path)
hi_data = load_json(hi_path)

# eng-kor
en_questions = [item["question"] for item in en_data]
en_answers = [item["answer"] for item in en_data]
ko_questions = [item["question"] for item in ko_data]
ko_answers = [item["answer"] for item in ko_data]

# eng-hindi
hi_questions = [item["question"] for item in hi_data]
hi_answers = [item["answer"] for item in hi_data]

# BERTScore eval (XLM-R base)
def calculate_bertscore(candidates, references, lang):
    P, R, F1 = score(candidates, references, lang=lang, model_type="xlm-roberta-large")
    return {
        "precision": float(P.mean().item()),
        "recall": float(R.mean().item()),
        "f1": float(F1.mean().item())
    }

# eng-kor eval
ko_question_scores = 0
ko_answer_scores = 0
ko_question_scores = calculate_bertscore(ko_questions, en_questions, "en")
ko_answer_scores = calculate_bertscore(ko_answers, en_answers, "en")

# eng-hindi eval
hi_question_scores = 0
hi_answer_scores = 0
hi_question_scores = calculate_bertscore(hi_questions, en_questions, "en")
hi_answer_scores = calculate_bertscore(hi_answers, en_answers, "en")

# Result dictionary
result = {
    "korean_translation": {
        "questions": ko_question_scores,
        "answers": ko_answer_scores
    },
    "hindi_translation": {
        "questions": hi_question_scores,
        "answers": hi_answer_scores
    }
}

# Output directory
save_dir = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "translation_eval_result.json")

# JSON file
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print(f"Evaluation Done: {save_path}")