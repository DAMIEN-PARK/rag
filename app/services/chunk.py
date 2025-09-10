from __future__ import annotations
from openai import OpenAI
from pathlib import Path
from typing import List, Tuple
import os, requests
import hashlib
import numpy as np
import json
import logging
# import fitz  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)


def extract_text_from_pdf_upstage(pdf_path: str | Path, api_key: str | None = None) -> str:
    api_key = api_key or os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY가 필요합니다.")

    pdf_path = Path(pdf_path)
    url = "https://api.upstage.ai/v1/document-ai/layout-analysis"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    with pdf_path.open("rb") as f:
        resp = requests.post(
            url,
            headers=headers,
            data={"ocr": False},
            files={"document": f},
        )

    if resp.status_code != 200:
        snippet = (resp.text or "")[:200]
        logger.error("Upstage API 응답 오류: %s", snippet)
        raise ValueError(f"예상치 못한 상태 코드: {resp.status_code}")

    try:
        data = resp.json()
    except ValueError as e:
        logger.error("Upstage API JSON 파싱 실패: %s", e)
        raise ValueError("Upstage API JSON 파싱 실패") from e

    elements = data.get("elements", [])
    texts = [el.get("text", "").strip() for el in elements if el.get("text")]
    text = "\n".join(t for t in texts if t).strip()
    if not text:
        logger.error("Upstage API에서 추출된 텍스트가 비어 있음: %s", pdf_path)
        raise ValueError("추출된 텍스트가 비어 있습니다.")

    return text



def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def get_embedding(text: str, dim: int = 1536) -> Tuple[List[float], str]:
    """주어진 텍스트에 대한 임베딩 벡터를 생성한다.

    If an OPENAI_API_KEY is present, use OpenAIEmbeddings via langchain.
    Otherwise, fall back to a deterministic pseudo-random vector without
    requiring external services.
    Returns a tuple of (vector, model_name).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다")

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vec = embeddings.embed_query(text)
    return vec, "openai"
