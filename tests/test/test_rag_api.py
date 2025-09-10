"""RAG API 엔드포인트 테스트."""
import os
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models import FakeListLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores.pgvector import PGVector
from app.api.v1.rag import router, get_rag_service
from app.services.rag.service import RAGService


def _create_service() -> RAGService:
    embedding = FakeEmbeddings(size=32)
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        store = PGVector(
            connection_string=database_url,
            embedding_function=embedding,
            collection_name="test_rag_api",
            pre_delete_collection=True,
        )
    else:
        store = InMemoryVectorStore(embedding=embedding)

    store.add_documents([Document(page_content="고양이는 귀엽다.")])
    llm = FakeListLLM(responses=["고양이는 정말 귀엽습니다."])
    return RAGService(store, llm)


def _create_app(service: RAGService) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_rag_service] = lambda: service
    return app


def test_query_returns_json():
    service = _create_service()
    app = _create_app(service)
    client = TestClient(app)

    res = client.post("/api/v1/rag/query", json={"question": "고양이는 어때?"})
    assert res.status_code == 200
    assert res.json() == {"answer": "고양이는 정말 귀엽습니다."}


def test_query_returns_text():
    service = _create_service()
    app = _create_app(service)
    client = TestClient(app)

    res = client.post(
        "/api/v1/rag/query",
        json={"question": "고양이는 어때?", "response_format": "text"},
    )
    assert res.status_code == 200
    assert res.text == "고양이는 정말 귀엽습니다."

