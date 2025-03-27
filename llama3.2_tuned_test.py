import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델 경로 설정
model_path = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU_Llamas/TOFU_Llama_FullFineTuning"

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# GPU 사용 가능 시 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 질문 입력 및 답변 생성 함수
def get_answer(question):
    # 입력 텍스트를 토큰화
    inputs = tokenizer(question, return_tensors="pt").to(device)
    
    # 모델로 답변 생성
    outputs = model.generate(
        **inputs,
        max_length=100,  # 최대 생성 길이
        num_return_sequences=1,  # 생성할 시퀀스 수
        temperature=0.7,  # 창의성 조절
        top_p=0.9,  # 누적 확률 기반 샘플링
        do_sample=True  # 샘플링 활성화
    )
    
    # 생성된 텍스트 디코딩
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 예시 질문
question = "Hina Ameen이 초기에 어떤 종류의 장르를 썼니?"
answer = get_answer(question)
print(f"Question: {question}")
print(f"Answer: {answer}")