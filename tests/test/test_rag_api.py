"""RAG API 엔드포인트 테스트."""
import os

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.api.v1.rag import router, get_rag_service
from app.services.rag.service import RAGService


def _create_service() -> RAGService:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")

    embedding = OpenAIEmbeddings()
    store = InMemoryVectorStore(embedding=embedding)
    store.add_documents([Document(page_content="고양이는 귀엽다.")])
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
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
    answer = res.json()["answer"]
    assert "고양" in answer
    assert "귀엽" in answer


def test_query_returns_text():
    service = _create_service()
    app = _create_app(service)
    client = TestClient(app)

    res = client.post(
        "/api/v1/rag/query",
        json={"question": "고양이는 어때?", "response_format": "text"},
    )
    assert res.status_code == 200
    assert "고양" in res.text
    assert "귀엽" in res.text

