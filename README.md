# RAG ChatBot 
 
- FastAPI
- LangChain/LangGraph
- PostgreSQL/pgvector (FAISS 미사용)

문서 인제스트 → 하이브리드 검색(BM25+벡터) → LLM 생성 응답까지 하나의 백엔드로 제공

**FastAPI, SQLAlchemy, Alembic, PostgreSQL(pgvector), LangChain/LangGraph** 를 사용합니다.

## 핵심 기능
* 파일 업로드→OCR→청킹→임베딩→pgvector 저장
* 하이브리드 검색: 키워드(BM25/Trigram) + 벡터 유사도
* Rerank 선택(BGE 등)
* 대화 세션 관리와 컨텍스트 응답
* 멀티 프로바이더(OpenAI/Anthropic/Google/Friendli EXAONE) 라우팅
* 평가 지표(hit\@k, MRR)

## 아키텍처 개요

```
[Upload] → loader/ocr → chunker → embed(router) → pgvector_store
                                      │
[Query] → retrieve(hybrid) → rerank? ─┤→ generate(llm/router) → Answer(+sources)
```

LangGraph 노드 흐름:

```
load_node → retrieve_node → (rerank_node) → generate_node
```

## 폴더 구조

프로젝트에 포함된 최신 구조입니다.

```
RAG/
├── app/                 FastAPI 애플리케이션
│   ├── main.py
│   ├── api/v1/          REST 엔드포인트(auth, files, rag, chat, admin)
│   ├── core/            설정·보안·로깅
│   ├── db/              Base/Session/Alembic seed
│   ├── models/          ORM(User/Project/Document/Chunk/Embedding/Conversation/Provider/LLM)
│   ├── schemas/         Pydantic 요청·응답
│   ├── crud/            DB 접근 계층
│   ├── services/        ingestion/vector/embed/rerank/llm/rag/chat
│   ├── langchain/       체인 예시
│   ├── langgraph/       그래프 노드·그래프
│   ├── prompts/         jinja2 프롬프트
│   ├── utils/           공통 유틸
│   └── middleware/      CORS·Rate limit
├── alembic/             마이그레이션
├── scripts/             ingest/rebuild/demo
├── docker/              Dockerfile·compose·nginx
├── configs/             logging·gunicorn
├── .env.example
├── pyproject.toml
├── alembic.ini
└── README.md
```
Endpoint Folder : `app/api/v1` 
## 요구 사항

* Python 3.12+
* Docker + Docker Compose
* PostgreSQL 16 with `pgvector`
* (선택) Tesseract OCR 또는 클라우드 OCR

## 빠른 시작

### 1) 환경 변수

`.env.example`를 복사해 `.env` 생성 후 값 설정:

```env
# DB(임시로컬)
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/postgres

# Auth
JWT_SECRET=change-me(임시)
ACCESS_TOKEN_EXPIRE_MIN=60

# Files
UPLOAD_DIR=./file/upload

# Embedding/LLM Providers
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
FRIENDLI_TOKEN=
FRIENDLI_TEAM_ID=
EXAONE_BASE_URL=https://api.friendli.ai/serverless/v1
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### 2) Docker로 실행

```bash
docker compose -f docker/docker-compose.yml up -d
# 최초 1회
alembic upgrade head
```

서버:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

Docs:

* Swagger: `http://localhost:5000/docs`
* ReDoc: `http://localhost:5000/redoc`

### 3) 로컬 개발(비도커)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
alembic upgrade head
uvicorn app.main:app --reload
```

## PostgreSQL 설정

pgvector 확장:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

벡터 컬럼 예시(1536차원):

```sql
-- models/embedding.py에 상응
ALTER TABLE chunk ADD COLUMN IF NOT EXISTS embedding vector(1536);
```

IVFFlat 인덱스 예시:

```sql
-- 검색 전 ANALYZE 필요
CREATE INDEX IF NOT EXISTS idx_chunk_embedding_ivf
ON chunk USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
ANALYZE chunk;
```

키워드 검색을 위한 Trigram:

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_chunk_content_trgm
ON chunk USING gin (content gin_trgm_ops);
```

## 사용법

### 파일 업로드 및 인제스트

```bash
curl -X POST http://localhost:5000/api/v1/files/upload \
  -F "file=@/path/doc.pdf" \
  -F "project_id=1"
```

응답 예:

```json
{ "document_id": 123, "status": "queued" }
```

### 질의(RAG)

```bash
curl -X POST http://localhost:5000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query":"퇴직금 산정 기준 설명",
    "project_id":1,
    "top_k":8,
    "use_rerank":true,
    "filters":{"doc_type":["pdf"],"uploader":["me"]}
  }'
```

응답 예:

```json
{
  "answer": "퇴직금은 ...",
  "sources": [
    {"document_id":123,"chunk_id":456,"score":0.82,"snippet":"..."},
    {"document_id":124,"chunk_id":400,"score":0.79,"snippet":"..."}
  ],
  "latency_ms": 842
}
```

### 대화

```bash
curl -X POST http://localhost:5000/api/v1/chat/session
# → { "session_id": "sess_..." }

curl -X POST http://localhost:5000/api/v1/chat/send \
  -H "Content-Type: application/json" \
  -d '{"session_id":"sess_...","message":"어제 올린 계약서 요약해줘"}'
```

## 평가 스크립트

```bash
python -m app.services.rag.evaluator --dataset path/to/queries.jsonl --top_k 8
# 출력: hit@k, MRR, nDCG 등
```

## 스크립트

* `scripts/ingest_file.py`: 단일 파일 CLI 인제스트
* `scripts/rebuild_index.py`: 인덱스 재생성
* `scripts/demo_query.py`: 간단 질의 데모

## 개발 팁

* 대용량 인덱스는 `ivfflat(lists=N)` 조절
* 하이브리드 검색: `hybrid_retriever.py`에서 가중치 조정(예: `α·BM25 + (1-α)·cosine`)
* Rerank는 비용 상승. 짧은 후보군(top\_k 20\~50)만 재랭크 권장
* OCR: Tesseract 성능 한계 시 클라우드 OCR로 폴백

## API 개략(예시)

* `POST /api/v1/files/upload`
* `POST /api/v1/rag/query`
* `POST /api/v1/chat/session`
* `POST /api/v1/chat/send`
* `GET  /api/v1/admin/models`
* `POST /api/v1/admin/providers`

스키마 예시

```ts
// QueryRequest
{
  query: string;
  project_id?: number;
  top_k?: number;          // default 8
  use_rerank?: boolean;    // default false
  filters?: Record<string,string[]>;
}

// AnswerResponse
{
  answer: string;
  sources: {document_id:number; chunk_id:number; score:number; snippet:string;}[];
  latency_ms?: number;
}
```

## 기술 스택
* **Backend**: FastAPI, Pydantic, SQLAlchemy, Alembic
* **DB**: PostgreSQL + pgvector, pg\_trgm
* **RAG**: LangChain, LangGraph
* **LLM/Embedding**: OpenAI, Anthropic, Google, Friendli(EXAONE), Hugging Face
* **Infra**: Docker, ~~Gunicorn~~, Nginx

## 로드맵
*우선 작업
* [ ] 멀티모달 인제스트(PDF 내 이미지+표 구조화)
* [ ] `*.확장자` pdf, jpg, txt, csv ... 판별하여 데이터 추출
---
* [ ]  우선순위 라우터(비용·지연·정확도 최적화)
* [ ] 세션 기반 장기 메모리 강화(vector_store)
* [ ] ~~벤치마크 스위트 및 리더보드~~

