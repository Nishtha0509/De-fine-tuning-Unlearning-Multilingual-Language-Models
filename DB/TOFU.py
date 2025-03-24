# from datasets import load_dataset
# from dotenv import load_dotenv
# import pandas as pd
# import os

# # .env 파일 로드
# load_dotenv()

# # 환경 변수에서 토큰 가져오기
# hf_token = os.getenv("HF_TOKEN")

# # Ensure the output directory exists
# output_dir = "DB/TOFU"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Load datasets
# dataset_full = load_dataset("locuslab/TOFU", "full")
# forget_01 = load_dataset("locuslab/TOFU", "forget01")
# forget_05 = load_dataset("locuslab/TOFU", "forget05")

# # Convert each dataset to pandas DataFrame
# dataset_full_df = pd.DataFrame(dataset_full['train'])
# forget_01_df = pd.DataFrame(forget_01['train'])
# forget_05_df = pd.DataFrame(forget_05['train'])

# # Save as valid JSON arrays (not line-delimited JSON)
# dataset_full_df.to_json(os.path.join(output_dir, 'full.json'), orient='records', indent=2, force_ascii=False)
# forget_01_df.to_json(os.path.join(output_dir, 'forget01.json'), orient='records', indent=2, force_ascii=False)
# forget_05_df.to_json(os.path.join(output_dir, 'forget05.json'), orient='records', indent=2, force_ascii=False)

# print("✅ JSON files have been saved in the 'TOFU' directory.")


from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd
import os
import requests
import json  # json 모듈 추가

# .env 파일 로드
load_dotenv()

# 환경 변수에서 토큰 가져오기
hf_token = os.getenv("HF_TOKEN")

# Ensure the output directory exists
output_dir = "TOFU"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ollama API 엔드포인트
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # 모델명 정확히 확인 필요

# 번역 함수 정의
def translate(text):
    """Ollama를 통해 번역 요청"""
    print(f"번역 중: {text[:30]}...")  # 번역할 문장의 처음 30자 출력
    prompt = f"다음을 한국어로 번역해줘:\n{text}"
    
    # API 요청 보내기
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        })
        
        # 응답 처리
        if response.status_code == 200:
            translated_text = response.json().get("response", "").strip()
            print(f"번역 완료: {translated_text[:30]}...")  # 번역된 문장의 앞부분 30자 출력
            return translated_text
        else:
            print(f"번역 오류 발생: {response.status_code}")
            return "번역 실패"
    except Exception as e:
        print(f"요청 중 오류 발생: {e}")
        return "번역 실패"

# Load datasets
print("데이터셋 로딩 중...")
dataset_full = load_dataset("locuslab/TOFU", "full")
forget_01 = load_dataset("locuslab/TOFU", "forget01")
forget_05 = load_dataset("locuslab/TOFU", "forget05")
print("데이터셋 로딩 완료.")

# Convert each dataset to pandas DataFrame
dataset_full_df = pd.DataFrame(dataset_full['train'])
forget_01_df = pd.DataFrame(forget_01['train'])
forget_05_df = pd.DataFrame(forget_05['train'])

# 번역 작업 및 파일 저장 함수
def translate_and_save(df, filename):
    """주어진 DataFrame을 번역 후 JSON 파일로 저장"""
    translated_data = []
    total_rows = len(df)
    print(f"총 {total_rows}개의 항목을 번역합니다...")

    for index, row in df.iterrows():
        print(f"진행 중: {index + 1}/{total_rows} ({(index + 1) / total_rows * 100:.2f}%)")
        
        # 번역
        translated_question = translate(row['question'])
        translated_answer = translate(row['answer'])
        
        # 결과 저장
        translated_data.append({
            "question": translated_question,
            "answer": translated_answer
        })
    
    # 번역된 데이터를 JSON 파일로 저장
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 번역 완료: {filename} 파일 저장 완료.")

# 번역 및 저장
print("번역을 시작합니다...")
# translate_and_save(dataset_full_df, 'full_kor.json')
translate_and_save(forget_01_df, 'forget01_kor.json')
# translate_and_save(forget_05_df, 'forget05_kor.json')

print("모든 파일이 'TOFU' 디렉토리에 저장되었습니다.")