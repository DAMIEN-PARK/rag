"""질의 → 검색 → 생성 엔드포인트."""

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from app.services.rag.service import RAGService, get_rag_service

router = APIRouter(prefix="/rag", tags=["rag"])


class QueryRequest(BaseModel):
    """사용자 질문 요청 바디."""

    question: str = Field(..., description="사용자 질문")
    response_format: str = Field(
        default="json", description="응답 형식: 'json' 또는 'text'"
    )


@router.post("/query")
async def query_rag(
    body: QueryRequest, service: RAGService = Depends(get_rag_service)
):
    """RAG 파이프라인을 통해 답변을 생성한다."""

    answer = service.query(body.question)
    if body.response_format == "text":
        return Response(content=answer, media_type="text/plain")
    return {"answer": answer}

