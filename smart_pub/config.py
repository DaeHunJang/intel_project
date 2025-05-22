import os
from pathlib import Path

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
RESULTS_DIR = PROJECT_ROOT / "results"

# 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 모델 후보 설정
MODEL_CANDIDATES = [
    "skt/ko-gpt-trinity-1.2B-v0.5",
    "EleutherAI/polyglot-ko-1.3b",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    "beomi/Llama-3-Open-Ko-8B"
]

# 기본 모델 설정
MODEL_ID = "EleutherAI/polyglot-ko-1.3b"
MODEL_QUANTIZATION = "int8"  # 옵션: none, int8, int4, gptq
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# RAG 설정
EMBEDDING_MODEL_ID = "jhgan/ko-sbert-nli"
VECTOR_DB_TYPE = "faiss"
RETRIEVAL_TOP_K = 3

# 감정 키워드 (20가지)
EMOTION_KEYWORDS = [
    "기분고조", "기분전환", "긴장완화", "깨끗함", "사교", "사색", 
    "상쾌함", "세련됨", "스트레스해소", "신뢰감", "안정", "여유", 
    "열정", "위로", "집중", "차분함", "특별한", "편안함", "행복", "활력"
]

# 감정-술 매핑
EMOTION_DRINK_MAPPING = {
    "기분고조": ["Fruity", "Sweet", "Colorful", "Celebratory"],
    "기분전환": ["Citrus", "Refreshing", "Herbal", "Sour"],
    "긴장완화": ["Smooth", "Warm", "Comforting", "Low alcohol"],
    "깨끗함": ["Clear", "Simple", "Pure", "Light"],
    "사교": ["Popular", "Easy to share", "Festive", "Medium alcohol"],
    "사색": ["Complex", "Bitter", "Aged", "Sipping"],
    "상쾌함": ["Sparkling", "Citrus", "Mint", "Light"],
    "세련됨": ["Classic", "Elegant", "Balanced", "Sophisticated"],
    "스트레스해소": ["Strong", "Bold", "Spicy", "High alcohol"],
    "신뢰감": ["Traditional", "Well-known", "Consistent", "Familiar"],
    "안정": ["Warm", "Smooth", "Comforting", "Mild"],
    "여유": ["Slow sipping", "Complex", "Rich", "Contemplative"],
    "열정": ["Spicy", "Red", "Bold", "Intense"],
    "위로": ["Sweet", "Creamy", "Warm", "Comforting"],
    "집중": ["Strong", "Pure", "Clear", "Direct"],
    "차분함": ["Subtle", "Balanced", "Herbal", "Low alcohol"],
    "특별한": ["Rare", "Unique", "Exotic", "Special occasion"],
    "편안함": ["Familiar", "Smooth", "Comforting", "Easy to drink"],
    "행복": ["Sweet", "Fruity", "Fun", "Colorful"],
    "활력": ["Energetic", "Citrus", "Sparkling", "Bright"]
}

# 프롬프트 템플릿
EMOTION_ANALYSIS_TEMPLATE = """
당신은 사용자의 감정을 분석하는 AI 어시스턴트입니다. 
사용자의 메시지에서 감정 상태를 분석하고, 다음 20가지 감정 키워드 중 가장 적합한 1-3개를 선택하세요:
{emotion_keywords}

사용자 메시지: {user_message}

감정 분석:
"""

DRINK_RECOMMENDATION_TEMPLATE = """
당신은 술 추천 전문 AI 바텐더입니다.
사용자의 감정 상태를 바탕으로 적합한 술을 추천해주세요.
사용자의 현재 감정: {emotion}

사용자가 이 감정 상태일 때 적합한 술을 하나 추천하고, 그 이유를 설명해주세요.
추천한 술에 대한 자세한 정보(재료, 특성 등)도 제공해주세요.

추천:
"""

CONVERSATION_TEMPLATE = """
당신은 Smart Pub의 AI 바텐더입니다. 
사용자의 감정에 맞는 술을 추천하고 관련 정보를 제공하는 역할을 합니다.
사용자와 자연스럽게 대화하면서 공감을 표현하고, 적절한 술을 추천해주세요.

대화 히스토리:
{conversation_history}

사용자: {user_message}

AI 바텐더:
"""

RAG_TEMPLATE = """
당신은 술에 대한 정보를 제공하는 AI 어시스턴트입니다.
다음 정보를 바탕으로 사용자의 질문에 답변하세요:

{context}

사용자 질문: {question}

답변:
"""