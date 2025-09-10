"""RAG 서비스 계층.

간단한 인메모리 벡터스토어와 LLM 체인을 조합하여
질문에 대한 답변을 생성한다. 실제 서비스에서는 적절한
벡터스토어 및 LLM 의존성을 주입해 사용한다.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models import (
    BaseLanguageModel,
    FakeListLLM,
)
from langchain_core.vectorstores import InMemoryVectorStore

from app.langchain.chains.qa_chain import build_qa_chain


class RAGService:
    """질의 응답을 수행하는 간단한 RAG 서비스."""

    def __init__(self, vectorstore: InMemoryVectorStore, llm: BaseLanguageModel):
        self.chain = build_qa_chain(vectorstore, llm)

    def query(self, question: str) -> str:
        """질문을 받아 답변 문자열을 반환한다."""
        return self.chain.invoke(question)


def _build_default_service() -> RAGService:
    """기본 벡터스토어와 LLM로 구성된 RAGService 생성."""
    embedding = FakeEmbeddings(size=32)
    vectorstore = InMemoryVectorStore(embedding=embedding)
    vectorstore.add_documents([Document(page_content="기본 문서입니다.")])
    llm = FakeListLLM(responses=["기본 응답입니다."])
    return RAGService(vectorstore, llm)


_default_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """FastAPI 의존성 주입을 위한 기본 서비스 제공자."""
    global _default_service
    if _default_service is None:
        _default_service = _build_default_service()
    return _default_service

