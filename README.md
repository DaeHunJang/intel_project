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
python download_operational_model.py
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

이 명령은 다음 모델들을 모두 평가하고 결과를 CSV로 저장합니다:
- skt/ko-gpt-trinity-1.2B-v0.5
- EleutherAI/polyglot-ko-1.3b
- beomi/Llama-3-OpenKoEn-8B
- naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B
- naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B

### 평가 결과 확인

```bash
python view_results.py
```

### 특정 모델만 평가

```bash
python run_evaluator.py --model-id beomi/Llama-3-OpenKoEn-8B
```

### 최적화 적용 후 평가

```bash
python run_evaluator.py --optimize int8
```

### 최적의 모델을 config에 자동 업데이트

```bash
python run_evaluator.py --update-config
```

## 모델 최적화

```bash
python optimize_models.py --method int8
```

지원하는 최적화 방법:
- `int8`: 8비트 정수 양자화
- `int4`: 4비트 정수 양자화
- `onnx`: ONNX 형식 변환

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
python main_recommender_app.py --model-id beomi/Llama-3-OpenKoEn-8B
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
├── build_vector_database.py
├── download_operational_model.py
├── main_recommender_app.py
├── models/
├── optimize_models.py
├── README.md
├── requirements.txt
├── results/
├── run_evaluator.py
├── smart_pub/
│   ├── __init__.py
│   ├── config.py
│   ├── conversation/
│   │   ├── __init__.py
│   │   └── dialogue_manager.py
│   ├── data/
│   │   └── drinks.json
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── drink_processor.py
│   ├── emotion_engine/
│   │   ├── __init__.py
│   │   └── emotion_analyzer.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── llm_wrapper.py
│   │   ├── model_evaluator.py
│   │   └── optimizer.py
│   ├── rag_engine/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── recommendation_engine/
│   │   ├── __init__.py
│   │   └── recommender.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── vector_db/
└── view_results.py
```

## 모델 후보

프로젝트에서 평가하는 한국어 LLM 모델:

1. **skt/ko-gpt-trinity-1.2B-v0.5**: 1.2B 파라미터의 한국어 특화 GPT 모델
2. **EleutherAI/polyglot-ko-1.3b**: 1.3B 파라미터의 한국어 특화 모델
3. **beomi/Llama-3-OpenKoEn-8B**: 8B 파라미터 크기의 Llama-3 기반 한국어-영어 이중 언어 모델
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