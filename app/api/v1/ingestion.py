from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import List, Dict
import os
from dotenv import load_dotenv

from app.schemas.ingestion import (
    UploadResponse, SplitRequest, SplitResponse,
    AnalyzeRequest, AnalyzeResponse,
    ExtractRequest, ExtractResponse,
    RunRequest, RunResponse)


# 모듈들은 사용자 분리 구조에 맞춰 import
from app.services.ingestion.preprocess.split_pdf import split_pdf
from app.services.ingestion.preprocess.analyzer_upstage import LayoutAnalyzer
from app.services.ingestion.preprocess.extract_assets import PDFImageProcessor

# render_html_md 가 별도면, PDFImageProcessor 내부에서 호출되도록 구성했거나 필요 시 아래 import 후 사용
# from app.services.ingestion.preprocess.render_html_md import render_html_and_md

load_dotenv(override=True)

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

STORAGE_ROOT = Path(os.getenv("INGESTION_STORAGE", "file/ingestion")).resolve()
UPLOAD_DIR = STORAGE_ROOT / "uploads"
ARTIFACT_DIR = STORAGE_ROOT / "artifacts"
for p in (UPLOAD_DIR, ARTIFACT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --- 파일 업로드 ---
@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF만 허용")
    # 경로 내 포함된 디렉터리 요소 제거
    filename = os.path.basename(file.filename)
    dest = (UPLOAD_DIR / filename).resolve()
    # 업로드 디렉터리 검증
    if not dest.is_relative_to(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="잘못된 파일 경로")

    # 중복 방지
    i = 1
    while dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        dest = (UPLOAD_DIR / f"{stem}_{i}{suffix}").resolve()
        i += 1

    if not dest.is_relative_to(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="잘못된 파일 경로")

    data = await file.read()
    dest.write_bytes(data)
    return UploadResponse(filename=dest.name, path=str(dest), size=len(data))

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
        json_path = analyzer.execute(str(p))
        out[str(p.resolve())] = str(Path(json_path).resolve())
    return AnalyzeResponse(json_paths=out)

# --- 자산 추출 + HTML/MD 렌더 ---
@router.post("/extract", response_model=ExtractResponse)
def extract_endpoint(req: ExtractRequest):
    pdf = Path(req.pdf_path)
    if not pdf.exists():
        raise HTTPException(status_code=404, detail="pdf_path가 존재하지 않음")
    proc = PDFImageProcessor(str(pdf))
    proc.extract_images()  # 이미지추출 메서드 실행 html/md 생성
    base = pdf.with_suffix("")  # 폴더
    images = sorted([str(p.resolve()) for p in base.glob("page_*_figure_*.png")])
    html_path = base / f"{base.name}.html"
    md_path = base / f"{base.name}.md"
    return ExtractResponse(
        output_folder=str(base.resolve()),
        images=images,
        html_path=str(html_path.resolve()),
        md_path=str(md_path.resolve()),
    )

# --- 전체 파이프라인 실행 ---
@router.post("/run", response_model=RunResponse)
def run_endpoint(req: RunRequest):
    # 1) split
    parts = split_pdf(req.pdf_path, batch_size=req.batch_size)
    # 2) analyze
    api_key = req.upstage_api_key or os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="UPSTAGE_API_KEY 필요")
    analyzer = LayoutAnalyzer(api_key)
    json_paths: Dict[str, str] = {}
    for part in parts:
        json_path = analyzer.execute(part)
        json_paths[str(Path(part).resolve())] = str(Path(json_path).resolve())
    # 3) extract + render
    ## proc <- PDF 처리기(Processor) 인스턴스 의 변수
    proc = PDFImageProcessor(req.pdf_path)
    proc.extract_images()
    base = Path(req.pdf_path).with_suffix("")
    images = sorted([str(p.resolve()) for p in base.glob("page_*_figure_*.png")])
    html_path = base / f"{base.name}.html"
    md_path = base / f"{base.name}.md"
    return RunResponse(
        parts=[str(Path(p).resolve()) for p in parts],
        json_paths=json_paths,
        output_folder=str(base.resolve()),
        html_path=str(html_path.resolve()),
        md_path=str(md_path.resolve()),
        images=images,
    )

# --- 산출물 바로 내려받기(선택) ---
@router.get("/artifact")
def get_artifact(path: str):
    f = Path(path)
    if not f.exists():
        raise HTTPException(status_code=404, detail="파일 없음")
    return FileResponse(str(f.resolve()))
