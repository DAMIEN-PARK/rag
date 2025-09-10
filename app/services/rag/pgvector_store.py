from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.db import crud
from app.db.session import SessionLocal


class PGVectorRetriever(BaseRetriever):
    """pgvector에 저장된 임베딩을 조회하는 간단한 Retriever."""

    embeddings: Embeddings
    k: int = 4

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)
        with SessionLocal() as db:
            chunks = crud.search_chunks_by_vector(db, query_vector, limit=self.k)
        return [Document(page_content=c.content, metadata={"chunk_id": c.id}) for c in chunks]

    async def _aget_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


class PGVectorStore:
    """`build_qa_chain`에 전달하기 위한 최소한의 인터페이스."""

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def as_retriever(self, search_kwargs: Optional[dict] = None) -> PGVectorRetriever:
        k = search_kwargs.get("k", 4) if search_kwargs else 4
        return PGVectorRetriever(embeddings=self.embeddings, k=k)
