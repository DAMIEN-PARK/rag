from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import os, requests
import hashlib
import numpy as np
import json
# import fitz  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        raise ValueError(f"예상치 못한 상태 코드: {resp.status_code}")

    output_file = pdf_path.with_suffix(".json")
    with output_file.open("w", encoding="utf-8") as out_f:
        json.dump(resp.json(), out_f, ensure_ascii=False, indent=2)

    return str(output_file)




    # api_key = api_key or os.getenv("UPSTAGE_API_KEY")
    # if not api_key:
    #     raise ValueError("UPSTAGE_API_KEY가 필요합니다.")
    #
    # url = "https://api.upstage.ai/v1/information-extraction"
    # headers = {"Authorization": f"Bearer {api_key}"}
    # with open(pdf_path, "rb") as f:
    #     resp = requests.post(
    #         url,
    #         headers=headers,
    #         files={"file": (Path(pdf_path).name, f, "application/pdf")},
    #     )
    # resp.raise_for_status()
    # data = resp.json()
    # text = data.get("text")
    # if text is None:
    #     raise ValueError(f"Upstage API 응답에 'text' 필드가 없습니다: {data}")
    # return text



def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def get_embedding(text: str, dim: int = 1536) -> Tuple[List[float], str]:
    """Generate an embedding vector for the given text.

    If an OPENAI_API_KEY is present, use OpenAIEmbeddings via langchain.
    Otherwise, fall back to a deterministic pseudo-random vector so that
    tests can run without external services.
    Returns a tuple of (vector, model_name).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(api_key=api_key)
            vec = embeddings.embed_query(text)
            return vec, "openai"
        except Exception:
            pass
    # fallback deterministic vector based on text hash
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    vec = rng.random(dim).tolist()
    return vec, "dummy"