# Smart Pub Project

**사용자의 감정 상태를 분석하여 맞춤형 술을 추천하는 대화형 애플리케이션**

## 프로젝트 개요

Smart Pub 앱은 LLM(대형 언어 모델)을 활용하여 사용자의 감정 상태를 분석하고 그에 맞는 술을 추천하는 대화형 애플리케이션입니다. 한국어 텍스트 분석에 특화된 다양한 LLM 모델을 활용하여 사용자의 감정을 정확하게 분석하고, RAG(Retrieval-Augmented Generation) 기법과 모델 최적화를 통해 효율적이고 정확한 추천 서비스를 제공합니다.

## 주요 기능

- **감정 분석**: 사용자 입력에서 20가지 감정 키워드 분류
- **맞춤형 술 추천**: 감정에 맞는 최적의 술 추천 및 설명
- **대화형 인터페이스**: 자연스러운 대화와 피드백 기반 추천 개선
- **RAG 기반 지식 제공**: 정확한 술 정보 검색 및 제공
- **최적화된 모델 추론**: 다양한 양자화 기법으로 성능 향상
- **자동 모델 평가**: 여러 모델 테스트 후 최적의 모델 선택

## 시작하기

### 필수 조건

- Python 3.8.10 이상
- GPU 지원 (권장, 최소 8GB VRAM)

### 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/smart_pub_app.git
cd smart_pub_app
```

2. 가상환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. 의존성 설치
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

4. 모델 다운로드
```bash
# 기본 모델 다운로드
python download_operational_model.py

# 임베딩 모델 다운로드
python download_operational_model.py --embedding

# 모든 후보 모델 다운로드
python download_operational_model.py --all
```

5. 데이터 처리 및 벡터 데이터베이스 구축
```bash
python build_vector_database.py
```

## 모델 평가 및 선택

### 모든 모델 평가 실행

```bash
python run_evaluator.py
```

이 명령은 다음 모델들을 모두 평가하고 결과를 저장합니다:
- skt/ko-gpt-trinity-1.2B-v0.5
- EleutherAI/polyglot-ko-1.3b
- beomi/Llama-3-Open-Ko-8B
- naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B
- naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B

### 평가 결과 확인

```bash
python view_results.py
```

### 특정 모델만 평가

```bash
python run_evaluator.py --model-id beomi/Llama-3-Open-Ko-8B
```

### 모델 정확도 테스트

```bash
# 모든 모델 정확도 테스트
python accuracy_test_alon.py --all

# 특정 모델 정확도 테스트
python accuracy_test_alon.py --model-id polyglot-ko-1.3b

# 자세한 결과 보기
python accuracy_test_alon.py --model-id polyglot-ko-1.3b --show-details
```

## 모델 최적화

```bash
# INT8 양자화
python optimize_models.py --method int8

# INT4 양자화
python optimize_models.py --method int4

# 특정 모델 최적화
python optimize_models.py --model-path ./models/polyglot-ko-1.3b --method int8
```

## 앱 실행

```bash
python main_recommender_app.py
```

추가 옵션:
```bash
# 디버그 모드 활성화
python main_recommender_app.py --debug

# 벡터 데이터베이스 재구축
python main_recommender_app.py --rebuild-vectordb

# 특정 모델 사용
python main_recommender_app.py --model-id beomi/Llama-3-Open-Ko-8B
```

## 대화 예시

```
사용자: 오늘 기분이 안좋아
AI 바텐더: 오늘 기분이 안좋으시군요. 기분 전환에 도움이 될 수 있는 '모히토'는 어떨까요? 
          상쾌한 라임과 민트의 조화가 스트레스 해소에 도움이 됩니다.

사용자: 좋아 어떤 재료가 들어갔어?
AI 바텐더: 모히토에는 라이트 럼, 라임 주스, 설탕, 민트 잎, 소다수가 들어갑니다. 
          특히 민트와 라임의 상쾌한 조합이 기분 전환에 좋은 영향을 줍니다.

사용자: 그거로 주문해줘
AI 바텐더: 모히토 주문이 완료되었습니다! 곧 준비해서 가져다 드리겠습니다. 
          시원하고 상쾌한 모히토와 함께 기분이 나아지길 바랍니다.
```

## 프로젝트 구조

```
smart_pub_project/
├── build_vector_database.py       # 벡터 DB 구축
├── download_operational_model.py  # 모델 다운로드
├── main_recommender_app.py        # 메인 애플리케이션
├── optimize_models.py             # 모델 최적화
├── run_evaluator.py               # 성능 평가
├── view_results.py                # 결과 시각화
├── requirements.txt               # 의존성 목록
├── smart_pub/                     # 메인 패키지
│   ├── config.py                  # 설정 파일
│   ├── conversation/              # 대화 관리
│   │   └── dialogue_manager.py
│   ├── data/                      # 데이터 파일
│   │   ├── drinks.json
│   │   └── test_cases.json
│   ├── data_processing/           # 데이터 처리
│   │   └── drink_processor.py
│   ├── emotion_engine/            # 감정 분석
│   │   └── emotion_analyzer.py
│   ├── model/                     # 모델 관리
│   │   ├── llm_wrapper.py
│   │   ├── model_evaluator.py
│   │   └── optimizer.py
│   ├── rag_engine/                # RAG 시스템
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── recommendation_engine/     # 추천 엔진
│   │   └── recommender.py
│   └── utils/                     # 유틸리티
│       └── helpers.py
├── models/                        # 다운로드된 모델들
├── vector_db/                     # 벡터 데이터베이스
└── results/                       # 평가 결과
```

## 모델 후보

프로젝트에서 평가하는 한국어 LLM 모델:

1. **skt/ko-gpt-trinity-1.2B-v0.5**: 1.2B 파라미터의 한국어 특화 GPT 모델
2. **EleutherAI/polyglot-ko-1.3b**: 1.3B 파라미터의 한국어 특화 모델
3. **beomi/Llama-3-Open-Ko-8B**: 8B 파라미터 크기의 Llama-3 기반 한국어-영어 이중 언어 모델
4. **naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B**: 네이버의 0.5B 경량 한국어 모델
5. **naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B**: 네이버의 1.5B 한국어 모델

## 평가 지표

| 지표 | 설명 |
|------|------|
| Latency | 모델 응답 생성 시간 |
| Throughput | 초당 처리 토큰 수 |
| Accuracy | 감정 분석 및 추천 정확도 |
| RAM Usage | 메모리 사용량 |
| GPU Usage | GPU 메모리 사용량 |

## 모델 최적화 기법

| 기법 | 설명 | 성능 향상 | 메모리 절감 |
|------|------|-----------|------------|
| INT8 양자화 | 8비트 정수 양자화 | 2-3x 빠른 추론 | ~75% 메모리 절감 |
| INT4 양자화 | 4비트 정수 양자화 | 3-4x 빠른 추론 | ~87.5% 메모리 절감 |
| ONNX 변환 | 크로스 플랫폼 추론 최적화 | 1.5-3x 빠른 추론 | 다양한 하드웨어 지원 |

## 감정 분석 및 매핑

프로젝트는 20가지 감정 키워드를 인식합니다:
- 기분고조, 기분전환, 긴장완화, 깨끗함, 사교, 사색
- 상쾌함, 세련됨, 스트레스해소, 신뢰감, 안정, 여유
- 열정, 위로, 집중, 차분함, 특별한, 편안함, 행복, 활력

각 감정은 술의 특성과 매핑되어 적절한 추천이 이루어집니다:
- **기분전환**: Citrus, Refreshing, Herbal 특성의 술
- **사교**: Popular, Easy to share, Festive 특성의 술
- **위로**: Sweet, Creamy, Warm, Comforting 특성의 술

## 성능 최적화

### 모델 양자화 전략

```python
# INT8 양자화 (권장)
BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# INT4 양자화 (메모리 중시)
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

### 벡터 데이터베이스

- **FAISS**: 고성능 벡터 유사도 검색
- **한국어 임베딩**: jhgan/ko-sbert-nli 모델 사용
- **검색 최적화**: L2 거리 기반 정확한 검색

## 트러블슈팅

### 모델 로드 실패
```bash
# 모델 다시 다운로드
python download_operational_model.py --model-id MODEL_NAME

# 로컬 모델 확인
python download_operational_model.py --list-models
```

### 벡터 데이터베이스 오류
```bash
# 벡터 DB 재구축
python build_vector_database.py

# 임베딩 모델 다운로드
python download_operational_model.py --embedding
```

### 메모리 부족
```bash
# 더 작은 모델 사용
python main_recommender_app.py --model-id HyperCLOVAX-SEED-Text-Instruct-0.5B

# 양자화 적용
python optimize_models.py --method int8
```
