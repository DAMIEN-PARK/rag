from fastapi import APIRouter
from . import auth, files, rag, chat_history, admin, ingestion, documents

router = APIRouter()
for r in (auth, files, rag, chat_history, admin, ingestion, documents):
    router.include_router(r.router)
