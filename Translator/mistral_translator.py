import os
import json
import time
import requests

os.environ["MISTRAL_API_KEY"] = "your_api"

API_URL = "https://api.mistral.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
    "Content-Type": "application/json",
}

def mistral_translate(text, target_language):
    """Translate text using Mistral AI's mistral-large-2407 model"""
    payload = {
        "model": "mistral-large-2407", 
        "messages": [{"role": "system", "content": f"Translate the following text to {target_language}. Maintain the original meaning and context: {text}"}],
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("Retrying ...")
        return mistral_translate(text, target_language)

def translate_json(input_path, output_path, target_language, batch_size=5):
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    translated_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        batch_translated = []
        for item in batch:
            translated_item = {
                "question": mistral_translate(item['question'], target_language),
                "answer": mistral_translate(item['answer'], target_language)
            }
            batch_translated.append(translated_item)
        
        translated_data.extend(batch_translated)
        
        print(f"Processed batch {i//batch_size + 1}")
        time.sleep(1) 
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(translated_data, file, ensure_ascii=False, indent=2)
    
    print(f"Translation to {target_language} completed. Saved to {output_path}")

input_path = r'/Users/nishtha/Desktop/Courses/CSE576_NLP/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full.json'
hindi_output_path = r'/Users/nishtha/Desktop/Courses/CSE576_NLP/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full_hindi.json'
korean_output_path = r'/Users/nishtha/Desktop/Courses/CSE576_NLP/full_kore.json'

# translate_json(input_path, korean_output_path, "Korean")
translate_json(input_path, hindi_output_path, "Hindi")
