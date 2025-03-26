import json
import time
import logging
import ollama
import requests
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('deepseek_translation_log.txt', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ollama 서버 설정
ollama_host = "http://sg020:11435"
try:
    response = requests.get(ollama_host)
    logger.info("Ollama server connected")
except requests.ConnectionError:
    logger.error("Ollama server not connected")
    raise

# Ollama 클라이언트 초기화
client = ollama.Client(host=ollama_host)

# 번역 함수 정의
def translate_text(text, target_language, max_retries=3):
    """
    Ollama를 통해 텍스트 번역
    
    :param text: 번역할 원본 텍스트
    :param target_language: 목표 언어
    :param max_retries: 최대 재시도 횟수
    :return: 번역된 텍스트
    """
    if not text or not isinstance(text, str):
        logger.warning(f"Invalid input for translation: {text}")
        return text

    # 번역 프롬프트
    prompt = f"Translate the following text to {target_language}. Maintain the original meaning and context precisely. Here is the text: '{text}'"
    
    for attempt in range(max_retries):
        try:
            # Ollama API 요청
            response = client.chat(
                model='deepseek-r1:671b',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # 응답 처리
            translated_text = response['message']['content'].strip()
            
            # 빈 응답 확인
            if not translated_text:
                logger.warning(f"Empty translation for text: {text[:100]}...")
                return text
            
            return translated_text
        
        except Exception as e:
            logger.error(f"Translation error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)  # 지수 백오프 대기
    
    # 모든 재시도 실패 시 원본 텍스트 반환
    logger.warning(f"Failed to translate text after {max_retries} attempts")
    return text

# JSON 파일 번역 함수
def translate_json(input_path, output_path, target_language, batch_size=10):
    """
    JSON 파일의 텍스트를 번역
    
    :param input_path: 입력 JSON 파일 경로
    :param output_path: 출력 JSON 파일 경로
    :param target_language: 목표 언어
    :param batch_size: 한 번에 처리할 항목 수
    """
    # 입력 파일 로드
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.info(f"Input file loaded. Total items: {len(data)}")
    
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading input file: {e}")
        return
    
    # 번역 프로세스
    translated_data = []
    try:
        # tqdm을 사용한 진행 상황 추적
        for i in tqdm(range(0, len(data), batch_size), desc=f"Translating to {target_language}"):
            batch = data[i:i+batch_size]
            
            # 배치 내 항목 번역
            batch_translated = []
            for item in batch:
                try:
                    translated_item = {
                        "question": translate_text(item.get('question', ''), target_language),
                        "answer": translate_text(item.get('answer', ''), target_language)
                    }
                    batch_translated.append(translated_item)
                except Exception as e:
                    logger.error(f"Error translating item: {e}")
            
            # 중간 결과 추가
            translated_data.extend(batch_translated)
            
            # 배치 사이 잠시 대기 (API 요청 제한 방지)
            time.sleep(5)
    
    except Exception as e:
        logger.critical(f"Translation process failed: {e}")
        return
    
    # 번역된 데이터 저장
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(translated_data, file, ensure_ascii=False, indent=2)
        
        logger.info(f"Translation to {target_language} completed. Saved to {output_path}")
        logger.info(f"Total translated items: {len(translated_data)}")
    
    except Exception as e:
        logger.error(f"Error saving translated data: {e}")

# 입력 및 출력 경로 설정
input_path = r'/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full.json'
korean_output_path = r'/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full_kor.json'
hindi_output_path = r'/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full_hindi.json'

# 번역 실행
if __name__ == "__main__":
    try:
        translate_json(input_path, korean_output_path, "Korean")
        translate_json(input_path, hindi_output_path, "Hindi")
    except Exception as e:
        logger.critical(f"Script execution failed: {e}")