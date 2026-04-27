import time
from pathlib import Path

from .classify import classify_tokens
from .engines import extract_tesseract, extract_vector, get_page_size_mm, load_gray_image
from .models import OCRResult


def run_ocr(
    input_path: str,
    engine: str = "auto",
    dpi: int = 400,
    conf_threshold: int = 30,
) -> OCRResult:
    start_time = time.time()
    ext = Path(input_path).suffix.lower()
    is_pdf = ext == ".pdf"

    if engine == "auto":
        engine = "vector" if is_pdf else "tesseract"

    page_width_mm, page_height_mm = get_page_size_mm(input_path, dpi=dpi)

    if engine == "vector":
        if not is_pdf:
            raise ValueError("vectorエンジンはPDFのみ対応です")
        tokens = extract_vector(input_path)
    else:
        tokens = extract_tesseract(
            load_gray_image(input_path, dpi=dpi),
            dpi=dpi,
            conf_threshold=conf_threshold,
        )

    tokens, dimensions, title_block = classify_tokens(
        tokens,
        page_width_mm,
        page_height_mm,
    )

    return OCRResult(
        tokens=tokens,
        dimensions=dimensions,
        title_block=title_block,
        engine=engine,
        elapsed_sec=time.time() - start_time,
        input_path=input_path,
    )