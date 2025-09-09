import pymupdf
from glob import glob
from PIL import Image
import os, requests, json, sys
from dotenv import load_dotenv
from pathlib import Path
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from markdownify import markdownify as markdown

load_dotenv(override=True)

sample_data = str(Path(__file__).resolve().parents[1] / "test.pdf")

def split_pdf(filepath, batch_size=10):
    """
    입력 PDF를 여러 개의 작은 PDF 파일로 분할
    """
    # PDF 파일 열기
    input_pdf = pymupdf.open(filepath)
    num_pages = len(input_pdf)
    print(f"총 페이지 수: {num_pages}")

    ret = []
    # PDF 분할
    for start_page in range(0, num_pages, batch_size):
        end_page = min(start_page + batch_size, num_pages) - 1

        # 분할된 PDF 저장
        input_file_basename = os.path.splitext(filepath)[0]
        output_file = f"{input_file_basename}_{start_page:04d}_{end_page:04d}.pdf"
        print(f"분할 PDF 생성: {output_file}")
        with pymupdf.open() as output_pdf:
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
            ret.append(output_file)

    # 입력 PDF 파일 닫기
    input_pdf.close()
    return ret

split_files = split_pdf(sample_data)

# Upstage Layout Analyzer
import os, requests, json, sys
from dotenv import load_dotenv

load_dotenv(override=True)


class LayoutAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key

    def _upstage_layout_analysis(self, input_file):
        """
         레이아웃 분석 API 호출

         :param input_file: 분석할 PDF 파일 경로
         :param output_file: 분석 결과를 저장할 JSON 파일 경로
        """
        input_file = str(Path(__file__).resolve().parents[1] / "test.pdf")
        output_file = "test.json"
        # API 요청 보내기
        response = requests.post(
            "https://api.upstage.ai/v1/document-ai/layout-analysis",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            data={"ocr": False},
            files={"document": open(input_file, "rb")},
        )

        # 응답 저장
        if response.status_code == 200:
            output_file = os.path.splitext(input_file)[0] + ".json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, ensure_ascii=False, indent=2)
            return output_file
        else:
            raise ValueError(f"예상치 못한 상태 코드: {response.status_code}")

    def execute(self, input_file):
        return self._upstage_layout_analysis(input_file)

analyzer = LayoutAnalyzer(os.environ.get("UPSTAGE_API_KEY"))
analyzed_files = []
for file in split_files:
    analyzed_files.append(analyzer.execute(file))



class PDFImageProcessor:
    """
    PDF 이미지 추출 및 HTML/Markdown 변환기
    """

    def __init__(self, pdf_file: str):
        """
        생성자
        :param pdf_file: PDF 파일 경로
        """
        self.pdf_file = pdf_file
        base = os.path.splitext(pdf_file)[0]
        # 같은 prefix를 가진 JSON 파일 자동 탐색
        self.json_files = sorted(glob(base + "*.json"))
        self.output_folder = base
        self.filename = os.path.basename(base)

    @staticmethod
    def _load_json(json_file: str):
        """JSON 파일 로드"""
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _get_page_sizes(json_data: dict):
        """메타데이터에서 각 페이지 크기 추출"""
        page_sizes = {}
        for page_element in json_data.get("metadata", {}).get("pages", []):
            width = page_element["width"]
            height = page_element["height"]
            page_num = page_element["page"]
            page_sizes[page_num] = [width, height]
        return page_sizes

    def pdf_to_image(self, page_num: int, dpi: int = 300):
        """PDF 페이지 → PIL 이미지 변환"""
        with fitz.open(self.pdf_file) as doc:
            page = doc[page_num - 1].get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [page.width, page.height], page.samples)
        return img

    @staticmethod
    def normalize_coordinates(coords: list, output_page_size: list):
        """좌표 정규화"""
        x_vals = [c["x"] for c in coords]
        y_vals = [c["y"] for c in coords]
        x1, y1, x2, y2 = min(x_vals), min(y_vals), max(x_vals), max(y_vals)

        w, h = output_page_size
        return (x1 / w, y1 / h, x2 / w, y2 / h)

    @staticmethod
    def crop_image(img: Image.Image, norm_coords: tuple, output_file: str):
        """좌표 기준 이미지 크롭"""
        x1, y1, x2, y2 = norm_coords
        W, H = img.size
        box = (int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
        cropped = img.crop(box)
        cropped.convert("RGB").save(output_file)

    def extract_images(self):
        """전체 파이프라인 실행"""
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"출력 폴더 생성: {self.output_folder}")

        figure_count = {}
        html_content = []

        for json_file in self.json_files:
            json_data = self._load_json(json_file)
            page_sizes = self._get_page_sizes(json_data)

            # 파일명에 페이지 범위 정보가 없을 경우 대비
            parts = os.path.basename(json_file).split("_")
            try:
                start_page = int(parts[1])
            except (IndexError, ValueError):
                start_page = 0

            for element in json_data.get("elements", []):
                if element.get("category") == "figure":
                    rel_page = element["page"]
                    page_num = start_page + rel_page

                    coords = element["bounding_box"]
                    output_size = page_sizes.get(rel_page, [612, 792])
                    pdf_img = self.pdf_to_image(page_num)
                    norm_coords = self.normalize_coordinates(coords, output_size)

                    # 페이지별 figure 번호
                    figure_count[page_num] = figure_count.get(page_num, 0) + 1

                    # 출력 파일명
                    output_file = os.path.join(
                        self.output_folder,
                        f"page_{page_num}_figure_{figure_count[page_num]}.png",
                    )
                    self.crop_image(pdf_img, norm_coords, output_file)
                    print(f"이미지 저장됨: {output_file}")

                    # HTML 업데이트
                    soup = BeautifulSoup(element["html"], "html.parser")
                    img_tag = soup.find("img")
                    if img_tag:
                        rel_path = os.path.relpath(output_file, self.output_folder)
                        img_tag["src"] = rel_path.replace("\\", "/")
                    element["html"] = str(soup)

                html_content.append(element.get("html", ""))

        # HTML 파일 저장
        html_path = os.path.join(self.output_folder, f"{self.filename}.html")
        combined_html = "\n".join(html_content)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(combined_html)
        print(f"HTML 저장 완료: {html_path}")

        # Markdown 저장
        md_path = os.path.join(self.output_folder, f"{self.filename}.md")
        md_out = markdown(combined_html)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_out)
        print(f"Markdown 저장 완료: {md_path}")

image_processor = PDFImageProcessor(sample_data)
image_processor.extract_images()


