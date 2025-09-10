from __future__ import annotations

"""문서를 파싱하고 임베딩하여 벡터스토어에 저장/로드하는 유틸리티."""

import json
import os
from pathlib import Path
from typing import List, Dict

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from .parser import parse_to_md_html
from .chunking import chunk_text
from .embedding import embed_text

# 벡터스토어 기본 경로 (환경 변수로 재정의 가능)
DEFAULT_VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", "file/vectorstore/faiss")).resolve()


def ingest_pdf_to_faiss(pdf_path: str, vs_path: Path | None = None) -> str:
    """PDF를 파싱하여 FAISS 벡터스토어로 저장한다.

    Args:
        pdf_path: 원본 PDF 경로.
        vs_path: 저장할 벡터스토어 경로. 주어지지 않으면 환경 변수를 사용한다.

    Returns:
        저장된 벡터스토어 경로 문자열.
    """
    vs_path = Path(vs_path or DEFAULT_VECTORSTORE_PATH).resolve()

    parse_out = parse_to_md_html(pdf_path)
    md_path = Path(parse_out["md_path"])
    with md_path.open("r", encoding="utf-8") as f:
        md_text = f.read()

    chunks = chunk_text(md_text)
    texts: List[str] = []
    vectors: List[List[float]] = []
    metadatas: List[Dict[str, str]] = []
    model_name = None
    for chunk in chunks:
        vec, model_name, _ = embed_text(chunk["text"])
        texts.append(chunk["text"])
        vectors.append(vec)
        metadatas.append({"source": pdf_path, "order": str(chunk["order"])})

    embedding_fn = OpenAIEmbeddings(model=model_name)
    vs = FAISS.from_embeddings(list(zip(texts, vectors)), embedding_fn, metadatas=metadatas)
    vs_path.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(vs_path))
    with (vs_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({"model": model_name}, f)
    return str(vs_path)


def load_faiss_vectorstore(vs_path: Path | None = None) -> FAISS:
    """저장된 FAISS 벡터스토어를 로드한다."""
    vs_path = Path(vs_path or DEFAULT_VECTORSTORE_PATH).resolve()
    meta_path = vs_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"메타데이터 파일이 존재하지 않습니다: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        model_name = json.load(f).get("model", "text-embedding-3-small")
    embedding_fn = OpenAIEmbeddings(model=model_name)
    return FAISS.load_local(str(vs_path), embedding_fn, allow_dangerous_deserialization=True)
