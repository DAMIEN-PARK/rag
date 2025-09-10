from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.langchain.chains.qa_chain import build_qa_chain
from app.services.rag.pgvector_store import PGVectorStore


class RAGService:
    """질의 응답을 수행하는 간단한 RAG 서비스."""

    def __init__(self, vectorstore: VectorStore, llm: BaseLanguageModel):
        self.chain = build_qa_chain(vectorstore, llm)

    def query(self, question: str) -> str:
        """질문을 받아 답변 문자열을 반환한다."""
        return self.chain.invoke(question)


def _build_default_service() -> RAGService:
    """PostgreSQL에 저장된 임베딩을 기반으로 RAGService를 생성한다."""

    load_dotenv()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = PGVectorStore(embeddings)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    return RAGService(vectorstore, llm)


_default_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """FastAPI 의존성 주입을 위한 기본 서비스 제공자."""
    global _default_service
    if _default_service is None:
        _default_service = _build_default_service()
    return _default_service
