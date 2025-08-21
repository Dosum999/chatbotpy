# 🚀 코디네이터 추천 RAG AI 챗봇 API 엔드포인트 정리

## 📋 프로젝트 개요
- **서비스명**: 코디네이터 추천 RAG AI 챗봇
- **버전**: 2.0.0
- **기술스택**: FastAPI, Google Gemini AI, RAG (Retrieval-Augmented Generation)
- **주요기능**: 요양보호사 코디네이터 추천, AI 응답 생성, 의미적 유사도 검색

---

## 🔌 API 엔드포인트 목록

### 1. **POST** `/chat` - RAG 챗봇 대화
**설명**: RAG 기술을 사용한 AI 챗봇과의 텍스트 대화

**요청 데이터**:
```json
{
  "message": "사용자 메시지"
}
```

**응답 데이터**:
```json
{
  "response": "AI 응답",
  "response_type": "response_type",
  "success": true,
  "error": null
}
```

**주요 기능**:
- 자연어 질의 처리
- RAG 기반 지식 검색
- 코디네이터 추천 로직
- 의미적 유사도 분석

---

### 2. **POST** `/voice-chat` - 음성 기반 챗봇
**설명**: 음성 입력을 통한 AI 챗봇 상호작용

**요청 데이터**:
```json
{
  "audio_data": "Base64 인코딩된 오디오 데이터",
  "audio_file": "오디오 파일 경로 (선택사항)"
}
```

**응답 데이터**:
```json
{
  "response": "AI 응답",
  "response_type": "response_type",
  "success": true,
  "error": null
}
```

**주요 기능**:
- 음성 인식 (Speech Recognition)
- 오디오 파일 처리
- 음성-텍스트 변환
- 텍스트 기반 AI 응답

---

### 3. **GET** `/` - 서비스 정보
**설명**: API 서비스 기본 정보 및 상태 확인

**응답 데이터**:
```json
{
  "service": "코디네이터 추천 RAG AI 챗봇",
  "version": "2.0.0",
  "status": "ready",
  "description": "RAG 기술을 사용한 요양보호사 코디네이터 추천 AI 챗봇 서비스입니다.",
  "features": [
    "RAG 검색",
    "AI 응답 생성", 
    "의미적 유사도 검색",
    "자연어 처리"
  ]
}
```

---

### 4. **GET** `/health` - 헬스 체크
**설명**: 서비스 상태 및 시스템 리소스 모니터링

**응답 데이터**:
```json
{
  "status": "healthy",
  "timestamp": 1234567890.123,
  "coordinators_loaded": 150,
  "vectorstore_ready": true,
  "documents_count": 150
}
```

**모니터링 항목**:
- 서비스 상태
- 코디네이터 데이터 로드 현황
- 벡터스토어 준비 상태
- 문서 개수

---

### 5. **POST** `/rebuild-vectorstore` - 벡터스토어 재구축
**설명**: RAG 검색을 위한 벡터스토어 재구축 및 업데이트

**응답 데이터**:
```json
{
  "message": "벡터스토어 재구축 완료",
  "coordinator_count": 150,
  "documents_count": 150,
  "success": true
}
```

**주요 기능**:
- 코디네이터 데이터 재로드
- 임베딩 벡터 재생성
- 벡터스토어 업데이트
- 시스템 최적화

---

## 🏗️ 데이터 모델 (Pydantic)

### ChatRequest
```python
class ChatRequest(BaseModel):
    message: str
```

### VoiceRequest
```python
class VoiceRequest(BaseModel):
    audio_data: Optional[str] = None  # Base64 인코딩된 오디오
    audio_file: Optional[str] = None  # 파일 경로
```

---

## 🔧 기술적 특징

### AI 모델
- **생성 모델**: Google Gemini 2.5 Flash
- **임베딩 모델**: Google Embedding-001
- **처리 방식**: RAG (Retrieval-Augmented Generation)

### 데이터베이스
- **DB 타입**: MySQL
- **연결**: SQLAlchemy + PyMySQL
- **테이블**: coordinators (코디네이터 정보)

### 벡터 검색
- **벡터스토어**: FAISS
- **유사도 계산**: Cosine Similarity
- **문서 분할**: RecursiveCharacterTextSplitter

---

## 📊 API 사용 통계

| 엔드포인트 | HTTP 메서드 | 용도 | 사용 빈도 |
|------------|-------------|------|-----------|
| `/chat` | POST | 메인 대화 기능 | ⭐⭐⭐⭐⭐ |
| `/voice-chat` | POST | 음성 대화 기능 | ⭐⭐⭐⭐ |
| `/health` | GET | 시스템 모니터링 | ⭐⭐⭐ |
| `/` | GET | 서비스 정보 | ⭐⭐ |
| `/rebuild-vectorstore` | POST | 시스템 관리 | ⭐ |

---

## 🚨 에러 처리

### 공통 에러 응답 형식
```json
{
  "response": "에러 메시지",
  "response_type": "error",
  "success": false,
  "error": "상세 에러 정보"
}
```

### 주요 에러 상황
- **API 키 누락**: GOOGLE_API_KEY 환경변수 미설정
- **DB 연결 실패**: MySQL 연결 오류
- **음성 처리 실패**: 오디오 파일 처리 오류
- **벡터스토어 오류**: 임베딩 생성 실패

---

## 🔒 보안 및 CORS

### CORS 설정
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 환경변수
- `GOOGLE_API_KEY`: Google AI API 키
- `DB_HOST`: 데이터베이스 호스트
- `DB_PORT`: 데이터베이스 포트
- `DB_NAME`: 데이터베이스 이름
- `DB_USER`: 데이터베이스 사용자
- `DB_PASSWORD`: 데이터베이스 비밀번호

---

## 📈 성능 최적화

### 캐싱 전략
- **코디네이터 캐시**: 메모리 기반 데이터 캐싱
- **벡터스토어**: FAISS 기반 고속 검색
- **임베딩 캐시**: 재계산 방지

### 검색 최적화
- **의미적 검색**: 임베딩 기반 유사도 계산
- **키워드 검색**: 텍스트 매칭
- **하이브리드 검색**: 의미적 + 키워드 결합

---

## 🎯 사용 시나리오

### 1. 일반 사용자
- 텍스트 기반 코디네이터 검색
- 자연어 질의를 통한 추천

### 2. 음성 사용자
- 음성 명령을 통한 검색
- 모바일 환경에서의 편의성

### 3. 시스템 관리자
- 헬스 체크를 통한 모니터링
- 벡터스토어 재구축으로 성능 최적화

---

## 🔮 향후 개발 계획

### 단기 계획
- [ ] API 응답 속도 개선
- [ ] 에러 로깅 시스템 강화
- [ ] API 문서 자동화 (Swagger)

### 장기 계획
- [ ] 실시간 채팅 기능
- [ ] 다국어 지원
- [ ] 고급 분석 대시보드
- [ ] 모바일 앱 연동

---

## 📞 기술 지원

### 개발팀 연락처
- **프로젝트**: 코디네이터 추천 RAG AI 챗봇
- **버전**: 2.0.0
- **상태**: Production Ready

### 문서 버전
- **작성일**: 2024년
- **최종수정**: 2024년
- **문서버전**: 1.0



