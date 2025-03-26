import json
import time
import google.generativeai as genai

# API 키 설정
API_KEY = ''  # 실제 API 키로 대체
genai.configure(api_key=API_KEY)

# 번역 함수 정의 (재시도 메커니즘 추가)
def translate_text(text, target_language, max_retries=3):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"Translate the following text to {target_language}. Maintain the original meaning and context: '{text}'"
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Translation error on attempt {attempt + 1}: {e}")
            if 'quota' in str(e).lower():
                # API 할당량 초과 시 대기 시간 추가
                wait_time = (attempt + 1) * 60  # 시도마다 대기 시간 증가
                print(f"Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                # 할당량 외 다른 오류는 즉시 반환
                return text
    
    # 모든 재시도 실패 시 원본 텍스트 반환
    return text

# JSON 파일 번역 함수 (배치 처리 추가)
def translate_json(input_path, output_path, target_language, batch_size=5):
    # JSON 파일 읽기
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 배치로 번역
    translated_data = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        # 배치 내 항목 번역
        batch_translated = []
        for item in batch:
            translated_item = {
                "question": translate_text(item['question'], target_language),
                "answer": translate_text(item['answer'], target_language)
            }
            batch_translated.append(translated_item)
        
        # 중간 결과 추가
        translated_data.extend(batch_translated)
        
        # 배치 사이 대기 시간
        print(f"Processed batch {i//batch_size + 1}")
        time.sleep(10)  # 각 배치 사이 10초 대기
    
    # 번역된 데이터 저장
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(translated_data, file, ensure_ascii=False, indent=2)
    
    print(f"Translation to {target_language} completed. Saved to {output_path}")

# 입력 및 출력 경로 설정
input_path = r'C:\Users\songj\OneDrive\Desktop\De-fine-tuning-Unlearning-Multilingual-Language-Models\DB\TOFU\full.json'
korean_output_path = r'C:\Users\songj\OneDrive\Desktop\De-fine-tuning-Unlearning-Multilingual-Language-Models\DB\TOFU\full_kor.json'
hindi_output_path = r'C:\Users\songj\OneDrive\Desktop\De-fine-tuning-Unlearning-Multilingual-Language-Models\DB\TOFU\full_hindi.json'

# 번역 실행
translate_json(input_path, korean_output_path, "Korean")
translate_json(input_path, hindi_output_path, "Hindi")