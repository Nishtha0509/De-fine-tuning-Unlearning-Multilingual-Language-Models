import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import torch
from tqdm import tqdm  # Added for better progress tracking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('translation_log.txt', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# NLLB 모델 초기화 (한번만 로드)
try:
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
    
    # Device selection with logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
except Exception as e:
    logger.error(f"Model initialization failed: {e}")
    raise

# 언어 코드 매핑
LANG_CODE_MAP = {
    "Korean": "kor_Hang",
    "Hindi": "hin_Deva"
}

# 번역 함수 (GPU 최적화 버전)
def translate_text(text, target_language):
    try:
        # Input validation
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid input for translation: {text}")
            return text
        
        lang_code = LANG_CODE_MAP.get(target_language)
        if not lang_code:
            logger.error(f"Unsupported target language: {target_language}")
            return text
        
        # 입력 텍스트 길이 제한 및 로깅
        if len(text) > 1024:
            logger.warning(f"Long input truncated. Original length: {len(text)}")
            text = text[:1024]
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():  # Explicitly disable gradient computation
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[lang_code],
                max_length=512,
                num_beams=5
            )
        
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    
    except Exception as e:
        logger.error(f"Translation error for text '{text[:100]}...': {e}")
        return text  # 실패 시 원문 반환

# JSON 번역 함수 (배치 처리 개선)
def translate_json(input_path, output_path, target_language):
    # Input validation
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Input file loaded. Total items: {len(data)}")
        
        # 전체 진행 상황 트래킹을 위한 tqdm 사용
        translated_data = []
        for item in tqdm(data, desc=f"Translating to {target_language}"):
            try:
                translated_item = {
                    "question": translate_text(item.get('question', ''), target_language),
                    "answer": translate_text(item.get('answer', ''), target_language)
                }
                translated_data.append(translated_item)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                # Optionally append original item or skip
        
        # 출력 디렉토리 확인
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Translation to {target_language} completed. Output: {output_path}")
        logger.info(f"Translated items: {len(translated_data)}")
    
    except Exception as e:
        logger.error(f"Translation process failed: {e}")

# 경로 설정
input_path = r'/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full.json'
korean_output_path = r'/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full_kor.json'
hindi_output_path = r'/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full_hindi.json'

# 번역 실행
try:
    translate_json(input_path, korean_output_path, "Korean")
    translate_json(input_path, hindi_output_path, "Hindi")
except Exception as e:
    logger.critical(f"Script execution failed: {e}")