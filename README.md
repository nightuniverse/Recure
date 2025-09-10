# 💊 Drug Repurposing Tool

AI 기반 약물 재목적화 시스템으로, 기존 약물을 새로운 질병 치료에 활용할 수 있는 후보를 찾아줍니다.

## 🎯 프로젝트 개요

이 프로젝트는 그래프 기반 분석과 텍스트 임베딩을 결합하여 약물-질병-유전자 관계를 분석하고, 새로운 약물 재목적화 후보를 제안합니다.

### 주요 기능
- **그래프 기반 분석**: NetworkX를 사용한 약물-질병-유전자 네트워크 분석
- **텍스트 임베딩**: Sentence Transformers를 활용한 의미적 유사도 계산
- **하이브리드 랭킹**: 그래프 점수와 텍스트 점수를 결합한 종합 평가
- **설명 가능한 AI**: 약물-질병 관계에 대한 상세한 근거 제시
- **웹 인터페이스**: Streamlit 기반 사용자 친화적 UI
- **REST API**: FastAPI 기반 확장 가능한 API 서비스

## 🚀 빠른 시작

### 1. 설치

```bash
# 의존성 설치
make install

# 또는 직접 설치
pip install -r requirements.txt
```

### 2. 실행

#### API 서버 실행
```bash
make run-api
# 또는
uvicorn api.main:app --reload --port 8000
```

#### 웹 애플리케이션 실행
```bash
make run-app
# 또는
streamlit run app/app.py
```

### 3. 테스트

```bash
make test
# 또는
pytest -q
```

## 📁 프로젝트 구조

```
Recure/
├── data/                    # 시드 데이터
│   ├── seed_drugs.csv
│   ├── seed_diseases.csv
│   ├── seed_drug_disease.csv
│   └── seed_drug_gene.csv
├── src/                     # 핵심 모듈
│   ├── data_loader.py      # 데이터 로더
│   ├── graph_builder.py    # 그래프 빌더
│   ├── text_embed.py       # 텍스트 임베딩
│   ├── ranker.py           # 랭킹 시스템
│   ├── explain.py          # 설명 생성
│   └── service.py          # 서비스 계층
├── api/                     # FastAPI 서버
│   └── main.py
├── app/                     # Streamlit UI
│   └── app.py
├── tests/                   # 테스트
│   └── test_pipeline.py
├── notebooks/               # 평가 노트북
│   └── eval_baseline.ipynb
├── requirements.txt         # 의존성
├── Makefile                # 빌드 스크립트
└── README.md
```

## 🔧 API 사용법

### 기본 엔드포인트

#### 1. 서비스 상태 확인
```bash
curl http://localhost:8000/health
```

#### 2. 약물 재목적화 후보 랭킹
```bash
curl "http://localhost:8000/rank?disease=Parkinson's%20disease&k=5"
```

#### 3. 약물-질병 관계 설명
```bash
curl "http://localhost:8000/explain?disease=Parkinson's%20disease&drug_id=D001"
```

#### 4. 모든 약물 목록
```bash
curl http://localhost:8000/drugs
```

#### 5. 모든 질병 목록
```bash
curl http://localhost:8000/diseases
```

### API 응답 예시

#### 랭킹 결과
```json
{
  "disease": "Parkinson's disease",
  "k": 5,
  "candidates": [
    {
      "drug_id": "D001",
      "drug_name": "metformin",
      "score": 0.8234,
      "text_score": 0.7891,
      "graph_score": 0.8765,
      "normalized_score": 1.0
    }
  ],
  "count": 5
}
```

#### 설명 결과
```json
{
  "drug_id": "D001",
  "drug_name": "metformin",
  "disease_id": "DI001",
  "disease_name": "Parkinson's disease",
  "graph_paths": [
    {
      "path_id": 1,
      "path": ["drug:D001", "gene:AMPK", "dis:DI001"],
      "explanation": "metformin targets AMPK → AMPK associated with Parkinson's disease"
    }
  ],
  "text_overlaps": {
    "overlapping_tokens": ["diabetes", "resistance"],
    "overlap_count": 2,
    "overlap_ratio": 0.15
  }
}
```

## 🧪 평가 및 테스트

### 단위 테스트 실행
```bash
pytest tests/ -v
```

### 성능 평가 (Jupyter 노트북)
```bash
jupyter notebook notebooks/eval_baseline.ipynb
```

평가 방법:
- **Leave-one-out 평가**: 알려진 약물-질병 관계를 숨기고 복구 능력 테스트
- **Hit@K 메트릭**: 상위 K개 후보 중 실제 약물 포함 비율

## 📊 데이터 구조

### 약물 데이터 (seed_drugs.csv)
- `drug_id`: 약물 고유 ID
- `drug_name`: 약물 이름
- `atc`: ATC 분류 코드
- `indications_text`: 적응증 텍스트

### 질병 데이터 (seed_diseases.csv)
- `disease_id`: 질병 고유 ID
- `disease_name`: 질병 이름
- `synonyms`: 동의어

### 관계 데이터
- `seed_drug_disease.csv`: 약물-질병 관계
- `seed_drug_gene.csv`: 약물-유전자 관계

## 🔬 기술 스택

- **Python 3.11**
- **FastAPI**: REST API 서버
- **Streamlit**: 웹 UI
- **NetworkX**: 그래프 분석
- **Sentence Transformers**: 텍스트 임베딩
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산
- **Pytest**: 테스트 프레임워크

## 🛠️ 개발 도구

### 코드 포맷팅
```bash
make format
# 또는
ruff check --fix .
black .
```

### 의존성 관리
```bash
pip install -r requirements.txt
```

## 🚧 로드맵

### 데이터 확장
- [ ] 더 큰 약물-질병 관계 데이터셋 통합
- [ ] 화학적 특성 데이터 추가
- [ ] 임상 시험 데이터 포함
- [ ] 유전자 발현 데이터 통합

### 모델 개선
- [ ] 그래프 임베딩 기법 도입 (Node2Vec, GraphSAGE)
- [ ] 더 정교한 텍스트 임베딩 모델 사용
- [ ] 앙상블 방법론 적용
- [ ] 딥러닝 기반 관계 예측 모델

### 시스템 개선
- [ ] 실시간 데이터 업데이트
- [ ] 분산 처리 지원
- [ ] 캐싱 시스템 도입
- [ ] 모니터링 및 로깅 강화

### 컴플라이언스
- [ ] 의료 데이터 보안 강화
- [ ] HIPAA 준수
- [ ] 감사 로그 시스템
- [ ] 데이터 거버넌스 정책

## 📝 라이선스

이 프로젝트는 연구 및 교육 목적으로 개발되었습니다.

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 nightskystaruniverse@kaist.ac.kr로 연락주세요.

---

**주의**: 이 도구는 연구 및 교육 목적으로만 사용되어야 하며, 실제 의료 결정에는 사용하지 마세요.
