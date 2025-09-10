from __future__ import annotations

from typing import Callable, List, Optional, Tuple
from pydantic import ConfigDict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sqlalchemy.orm import Session

from app.db import crud


class PGVectorRetriever(BaseRetriever):
    db: Session
    embed_fn: Callable[[str], Tuple[List[float], str]]
    k: int
    model_config = ConfigDict(arbitrary_types_allowed=True)


    def _get_relevant_documents(self, query: str, run_manager=None) -> List[Document]:
        vector, _ = self.embed_fn(query)
        results = crud.search_chunks_by_vector(self.db, vector, self.k)
        return [
            Document(
                page_content=chunk.content,
                metadata={
                    "distance": distance,
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                },
            )
            for chunk, distance in results
        ]

    async def _aget_relevant_documents(self, query: str, run_manager=None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager)


class PGVectorStore:
    def __init__(self, db: Session, embed_fn: Callable[[str], Tuple[List[float], str]]):
        self.db = db
        self.embed_fn = embed_fn

    def as_retriever(self, search_kwargs: Optional[dict] = None) -> PGVectorRetriever:
        k = 4
        if search_kwargs and "k" in search_kwargs:
            k = search_kwargs["k"]
        return PGVectorRetriever(db=self.db, embed_fn=self.embed_fn, k=k)

    def search(self, query: str, k: int = 4) -> List[Document]:
        """주어진 질의와 가장 유사한 청크들을 반환한다."""
        vector, _ = self.embed_fn(query)
        results = crud.search_chunks_by_vector(self.db, vector, k)
        return [
            Document(
                page_content=chunk.content,
                metadata={
                    "distance": distance,
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                },
            )
            for chunk, distance in results
        ]
