from __future__ import annotations
from typing import TypedDict, Literal, Optional, Dict, Any
from pathlib import Path
import json, os

# 충돌  문제 해결하기 위해서 fitz 강제
import pymupdf as fitz
from PIL import Image
from bs4 import BeautifulSoup

BlockType = Literal["text", "table", "figure"]


class Block(TypedDict, total=False):
    text: str
    block_type: BlockType
    page_num: int
    coords: Dict[str, float]
    caption: Optional[str]
    html: Optional[str]


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _page_sizes(j: Dict[str, Any]) -> Dict[int, list[float]]:
    sizes: Dict[int, list[float]] = {}
    for p in j.get("metadata", {}).get("pages", []):
        sizes[p["page"]] = [p["width"], p["height"]]
    return sizes


def _norm_bbox(
    coords: list[Dict[str, float]], wh: list[float]
) -> tuple[float, float, float, float]:
    xs = [c["x"] for c in coords]
    ys = [c["y"] for c in coords]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w, h = wh
    return (x1 / w, y1 / h, x2 / w, y2 / h)


def _crop(
    img: Image.Image, nb: tuple[float, float, float, float], out_path: str | Path
) -> None:
    x1, y1, x2, y2 = nb
    W, H = img.size
    box = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
    img.crop(box).convert("RGB").save(out_path)


# def pdf_page_to_image(pdf_path: str | Path, page_num_1based: int, dpi: int = 300) -> Image.Image:
#     images = convert_from_path(str(pdf_path), dpi=dpi,
#                                first_page=page_num_1based, last_page=page_num_1based)
#     if not images:
#         raise ValueError(f"페이지 {page_num_1based} 렌더 실패")
#     return images[0]


def pdf_page_to_image(
    pdf_path: str | Path, page_num_1based: int, dpi: int = 300
) -> Image.Image:
    with fitz.open(pdf_path) as doc:
        page = doc[page_num_1based - 1]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        return img.convert("RGB")


def extract_blocks_and_images(
    pdf_path: str | Path,
    json_paths: list[str | Path],
    out_dir: str | Path,
) -> tuple[list[Block], list[str]]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_images: list[str] = []
    blocks: list[Block] = []

    for jp in sorted(json_paths):
        j = _load_json(jp)
        sizes = _page_sizes(j)

        name_parts = Path(jp).stem.split("_")
        try:
            start_page = int(name_parts[-2])
        except Exception:
            start_page = 0

        for el in j.get("elements", []):
            cat = el.get("category")
            rel_page = int(el.get("page", 1))
            abs_page = start_page + rel_page

            b: Block = {
                "block_type": "text" if cat not in ("figure", "table") else cat,
                "page_num": abs_page,
            }

            text = el.get("text")
            if text:
                b["text"] = text

            caption = el.get("caption")
            if caption:
                b["caption"] = caption

            coords_list = el.get("bounding_box")
            if coords_list:
                nb = _norm_bbox(coords_list, sizes.get(rel_page, [612, 792]))
                b["coords"] = {"x1": nb[0], "y1": nb[1], "x2": nb[2], "y2": nb[3]}
            else:
                nb = None

            html = el.get("html")
            if cat == "figure" and nb:
                img = pdf_page_to_image(pdf_path, abs_page)
                idx = len([p for p in saved_images if f"page_{abs_page}_" in p]) + 1
                out_img = out_dir / f"page_{abs_page}_figure_{idx}.png"
                _crop(img, nb, out_img)
                saved_images.append(str(out_img))

                if html:
                    soup = BeautifulSoup(html, "html.parser")
                    img_tag = soup.find("img")
                    if img_tag:
                        rel_path = os.path.relpath(out_img, out_dir)
                        img_tag["src"] = rel_path.replace("\\", "/")
                    html = str(soup)

            if html:
                b["html"] = html

            blocks.append(b)
    return blocks, saved_images
