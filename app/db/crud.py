from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models
from app.schemas.db import (
    FileCreate,
    DocumentCreate,
    ChunkCreate,
    EmbeddingCreate,
    ChatHistoryCreate,
)


def _commit(db: Session) -> None:
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


def create_file(db: Session, file_in: FileCreate) -> models.File:
    db_obj = models.File(**file_in.model_dump())
    db.add(db_obj)
    _commit(db)
    db.refresh(db_obj)
    return db_obj


def get_file(db: Session, file_id: int) -> models.File | None:
    return db.query(models.File).filter(models.File.id == file_id).first()


def list_files(db: Session) -> list[models.File]:
    return db.query(models.File).all()


def create_document(db: Session, doc_in: DocumentCreate) -> models.Document:
    db_obj = models.Document(**doc_in.model_dump())
    db.add(db_obj)
    _commit(db)
    db.refresh(db_obj)
    return db_obj


def get_document(db: Session, doc_id: int) -> models.Document | None:
    return db.query(models.Document).filter(models.Document.id == doc_id).first()


def list_documents(db: Session) -> list[models.Document]:
    return db.query(models.Document).all()


def create_chunk(db: Session, chunk_in: ChunkCreate) -> models.Chunk:
    db_obj = models.Chunk(**chunk_in.model_dump())
    db.add(db_obj)
    _commit(db)
    db.refresh(db_obj)
    return db_obj


def list_chunks_by_document(db: Session, document_id: int) -> list[models.Chunk]:
    return (
        db.query(models.Chunk)
        .filter(models.Chunk.document_id == document_id)
        .order_by(models.Chunk.chunk_order)
        .all()
    )


def create_embedding(db: Session, emb_in: EmbeddingCreate) -> models.Embedding:
    db_obj = models.Embedding(**emb_in.model_dump())
    db.add(db_obj)
    _commit(db)
    db.refresh(db_obj)
    return db_obj


def create_chathistory(db: Session, log_in: ChatHistoryCreate) -> models.ChatHistory:
    db_obj = models.ChatHistory(**log_in.model_dump())
    db.add(db_obj)
    _commit(db)
    db.refresh(db_obj)
    return db_obj

def list_chathistory(db: Session) -> list[models.ChatHistory]:
    return db.query(models.ChatHistory).order_by(models.ChatHistory.created_at.desc()).all()


def search_chunks_by_vector(
    db: Session, query_vector: list[float], top_k: int = 5
) -> list[tuple[models.Chunk, float]]:
    """주어진 벡터와 가장 유사한 청크를 pgvector를 사용해 검색한다."""
    stmt = (
        select(
            models.Chunk,
            models.Embedding.vector.cosine_distance(query_vector).label("distance"),
        )
        .join(models.Embedding, models.Chunk.id == models.Embedding.chunk_id)
        .order_by(models.Embedding.vector.cosine_distance(query_vector))
        .limit(top_k)
    )
    return [
        (chunk, float(distance))
        for chunk, distance in db.execute(stmt).all()
    ]
