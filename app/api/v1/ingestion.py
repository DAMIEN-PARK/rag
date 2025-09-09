from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from shutil import copyfileobj
from starlette.concurrency import run_in_threadpool
from pathlib import Path
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db import crud
from app.schemas.db import (
    FileCreate,
    DocumentCreate,
    ChunkCreate,
    EmbeddingCreate,
)

from app.schemas.ingestion import (
    UploadResponse, SplitRequest, SplitResponse,
    AnalyzeRequest, AnalyzeResponse,
    ExtractRequest, ExtractResponse,
    RunRequest, RunResponse)

# 모듈들은 사용자 분리 구조에 맞춰 import
from app.services.ingestion.preprocess.split_pdf import split_pdf
from app.services.ingestion.preprocess.analyzer_upstage import LayoutAnalyzer
from app.services.ingestion.preprocess.extract_assets import extract_blocks_and_images
from app.services.ingestion.preprocess.render_html_md import render_all
from app.services.chunk import (
    extract_text_from_pdf_upstage,
    chunk_text,
    get_embedding,
)

load_dotenv(override=True)

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

STORAGE_ROOT = Path(os.getenv("INGESTION_STORAGE", "file/ingestion")).resolve()
UPLOAD_DIR = STORAGE_ROOT / "uploads"
ARTIFACT_DIR = STORAGE_ROOT / "artifacts"
for p in (UPLOAD_DIR, ARTIFACT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --- 파일 업로드 ---
@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF만 허용")
    dest = (UPLOAD_DIR / file.filename).resolve()
    if not dest.is_relative_to(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="업로드 경로 밖의 파일은 허용되지 않습니다")
    # 중복 방지
    i = 1
    while dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        dest = (UPLOAD_DIR / f"{stem}_{i}{suffix}").resolve()
        if not dest.is_relative_to(UPLOAD_DIR):
            raise HTTPException(status_code=400, detail="업로드 경로 밖의 파일은 허용되지 않습니다")
        i += 1
    with dest.open("wb") as out_file:
        await run_in_threadpool(copyfileobj, file.file, out_file)
    db_file = crud.create_file(
        db,
        FileCreate(
            original_name=file.filename,
            mime_type=file.content_type,
            storage_path=str(dest),
        ),
    )
    return UploadResponse(
        filename=dest.name,
        path=str(dest),
        size=dest.stat().st_size,
        file_id=db_file.id,
    )

# --- PDF 분할 ---
@router.post("/split", response_model=SplitResponse)
def split_endpoint(req: SplitRequest):
    pdf = Path(req.pdf_path)
    if not pdf.exists():
        raise HTTPException(status_code=404, detail="pdf_path가 존재하지 않음")
    parts = split_pdf(str(pdf), batch_size=req.batch_size)
    return SplitResponse(parts=[str(Path(p).resolve()) for p in parts])

# --- Upstage Layout 분석 ---
@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(req: AnalyzeRequest):
    api_key = req.upstage_api_key or os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="UPSTAGE_API_KEY 필요")
    analyzer = LayoutAnalyzer(api_key)
    out: Dict[str, str] = {}
    for pdf_path in req.pdf_paths:
        p = Path(pdf_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"없음: {pdf_path}")
        try:
            json_path = analyzer.execute(str(p))
        except ValueError as e:
            raise HTTPException(status_code=502, detail=str(e))
        out[str(p.resolve())] = str(Path(json_path).resolve())
    return AnalyzeResponse(json_paths=out)

# --- 자산 추출 + HTML/MD 렌더 ---
@router.post("/extract", response_model=ExtractResponse)
def extract_endpoint(req: ExtractRequest):
    pdf = Path(req.pdf_path)
    if not pdf.exists():
        raise HTTPException(status_code=404, detail="pdf_path가 존재하지 않음")
    # 산출물은 ARTIFACT_DIR/<원본파일명>/ 구조로 저장
    base = ARTIFACT_DIR / pdf.stem

    json_paths = sorted(pdf.parent.glob(f"{pdf.stem}_*.json"))
    if not json_paths:
        raise HTTPException(status_code=404, detail="연관된 JSON 파일이 없습니다")
    blocks, images = extract_blocks_and_images(
        pdf_path=str(pdf),
        json_paths=[str(p) for p in json_paths],
        out_dir=str(base),
    )
    render_result = render_all(blocks, out_dir=str(base), base_name=pdf.stem)
    images = [str(Path(p).resolve()) for p in images]
    return ExtractResponse(
        output_folder=str(base.resolve()),
        images=images,
        html_path=str(Path(render_result["html_path"]).resolve()),
        md_path=str(Path(render_result["md_path"]).resolve()),
        blocks=blocks,
    )

# --- 전체 파이프라인 실행 ---
@router.post("/run", response_model=RunResponse)
def run_endpoint(req: RunRequest, db: Session = Depends(get_db)):
    pdf = Path(req.pdf_path)
    if not pdf.exists():
        raise HTTPException(status_code=404, detail="pdf_path가 존재하지 않음")

    db_file = crud.create_file(
        db,
        FileCreate(
            original_name=pdf.name,
            mime_type="application/pdf",
            storage_path=str(pdf.resolve()),
        ),
    )
    parts = split_pdf(str(pdf), batch_size=req.batch_size)

    api_key = req.upstage_api_key or os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="UPSTAGE_API_KEY 필요")
    analyzer = LayoutAnalyzer(api_key)
    json_paths: Dict[str, str] = {}
    documents: list[tuple[str, int]] = []
    for part in parts:
        try:
            json_path = analyzer.execute(part)
        except ValueError as e:
            raise HTTPException(status_code=502, detail=str(e))
        json_paths[str(Path(part).resolve())] = str(Path(json_path).resolve())

        doc = crud.create_document(
            db,
            DocumentCreate(
                file_id=db_file.id,
                title=Path(part).name,
                doc_meta={"pdf_path": str(Path(part).resolve()), "json_path": str(Path(json_path).resolve())},
            ),
        )
        documents.append((part, doc.id))

    for part, doc_id in documents:
        text = extract_text_from_pdf_upstage(part)
        pieces = chunk_text(text)
        for order, piece in enumerate(pieces, start=1):
            chunk_db = crud.create_chunk(
                db,
                ChunkCreate(
                    document_id=doc_id,
                    content=piece,
                    chunk_order=order,
                ),
            )
            vec, model_name = get_embedding(piece)
            crud.create_embedding(
                db,
                EmbeddingCreate(
                    chunk_id=chunk_db.id,
                    vector=vec,
                    model=model_name,
                    dim=len(vec),
                ),
            )

    base = ARTIFACT_DIR / pdf.stem
    blocks, images = extract_blocks_and_images(
        pdf_path=str(pdf),
        json_paths=list(json_paths.values()),
        out_dir=str(base),
    )
    render_result = render_all(blocks, out_dir=str(base), base_name=base.name)
    images = [str(Path(p).resolve()) for p in images]
    md_path = Path(render_result["md_path"])
    html_path = Path(render_result["html_path"])

    # 4) chunking & embedding
    with md_path.open("r", encoding="utf-8") as f:
        md_text = f.read()
    chunks = chunk_text(md_text)
    chunk_texts = [c["text"] for c in chunks]
    embeddings: List[List[float]] = []
    model_name: Optional[str] = None
    dim: Optional[int] = None
    for c in chunk_texts:
        vec, model_name = get_embedding(c)
        dim = len(vec)
        embeddings.append(vec)

    return RunResponse(
        parts=[str(Path(p).resolve()) for p in parts],
        json_paths=json_paths,
        output_folder=str(base.resolve()),
        html_path=str(html_path.resolve()),
        md_path=str(md_path.resolve()),
        images=images,
        blocks=blocks,
        chunks=chunk_texts,
        embeddings=embeddings,
        embedding_model=model_name,
        embedding_dim=dim,
    )

# --- 산출물 바로 내려받기(추후 필요하면) ---
@router.get("/artifact")
def get_artifact(path: str):
    f = Path(path)
    if not f.exists():
        raise HTTPException(status_code=404, detail="파일 없음")
    return FileResponse(str(f.resolve()))
