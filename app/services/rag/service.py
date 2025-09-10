from __future__ import annotations
from typing import Optional
# from langchain_core.documents import Document
# from langchain_core.embeddings import FakeEmbeddings
# from langchain_core.language_models import (
#     BaseLanguageModel,
#     FakeListLLM,
# )
# from langchain_core.vectorstores import InMemoryVectorStore
import os
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

from app.langchain.chains.qa_chain import build_qa_chain


class RAGService:
    """질의 응답을 수행하는 간단한 RAG 서비스."""

    def __init__(self, vectorstore: VectorStore, llm: BaseLanguageModel):
        self.chain = build_qa_chain(vectorstore, llm)

    def query(self, question: str) -> str:
        """질문을 받아 답변 문자열을 반환한다."""
        return self.chain.invoke(question)


def _build_default_service() -> RAGService:
    """기본 PGVector 저장소와 OpenAI 기반 LLM로 구성된 RAGService 생성."""

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL 환경 변수가 설정되어 있지 않습니다.")

    embedding = OpenAIEmbeddings()
    vectorstore = PGVector(
        connection_string=database_url,
        embedding_function=embedding,
        collection_name="rag_default",
    )
    llm = ChatOpenAI()
    return RAGService(vectorstore, llm)


_default_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """FastAPI 의존성 주입을 위한 기본 서비스 제공자."""
    global _default_service
    if _default_service is None:
        _default_service = _build_default_service()
    return _default_service

