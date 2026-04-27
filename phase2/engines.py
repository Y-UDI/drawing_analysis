from pathlib import Path

import cv2
import numpy as np

from .models import PT2MM, TextToken


def _import_pymupdf():
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF が必要です: pip install pymupdf") from exc
    return fitz


def extract_vector(pdf_path: str, page_no: int = 0) -> list:
    fitz = _import_pymupdf()
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    tokens = []

    span_map = {}
    raw = page.get_text("dict")
    for block in raw["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                cx = (span["bbox"][0] + span["bbox"][2]) / 2
                cy = (span["bbox"][1] + span["bbox"][3]) / 2
                span_map[(round(cx, 0), round(cy, 0))] = span["size"]

    words = page.get_text("words")
    for word in words:
        x0, y0, x1, y1, text = word[0], word[1], word[2], word[3], word[4]
        text = text.strip()
        if not text:
            continue

        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        best_fs, best_dist = 10.0, 1e9
        for (sx, sy), font_size in span_map.items():
            distance = abs(sx - cx) + abs(sy - cy)
            if distance < best_dist:
                best_dist, best_fs = distance, font_size

        tokens.append(TextToken(
            text=text,
            x=cx * PT2MM,
            y=cy * PT2MM,
            x0=x0 * PT2MM,
            y0=y0 * PT2MM,
            x1=x1 * PT2MM,
            y1=y1 * PT2MM,
            font_size=best_fs,
            confidence=1.0,
            source="vector",
        ))

    doc.close()
    return tokens


def preprocess_for_ocr(img_gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=7)
    height, width = denoised.shape
    upscaled = cv2.resize(
        denoised,
        (width * 2, height * 2),
        interpolation=cv2.INTER_LANCZOS4,
    )
    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)


def extract_tesseract(
    img_gray: np.ndarray,
    dpi: int = 400,
    conf_threshold: int = 30,
) -> list:
    try:
        import pytesseract
    except ImportError as exc:
        raise ImportError("pip install pytesseract && apt install tesseract-ocr") from exc

    preprocessed = preprocess_for_ocr(img_gray)
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(preprocessed)
    config = "--psm 12 --oem 3"
    data = pytesseract.image_to_data(
        pil_img,
        config=config,
        lang="eng",
        output_type=pytesseract.Output.DICT,
    )

    px2mm = 25.4 / dpi / 2
    tokens = []

    for index, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue

        conf_raw = data["conf"][index]
        if not str(conf_raw).lstrip("-").isdigit():
            continue

        conf = int(conf_raw)
        if conf < conf_threshold:
            continue

        x_px = data["left"][index]
        y_px = data["top"][index]
        w_px = data["width"][index]
        h_px = data["height"][index]

        tokens.append(TextToken(
            text=text,
            x=(x_px + w_px / 2) * px2mm,
            y=(y_px + h_px / 2) * px2mm,
            x0=x_px * px2mm,
            y0=y_px * px2mm,
            x1=(x_px + w_px) * px2mm,
            y1=(y_px + h_px) * px2mm,
            font_size=h_px * px2mm * 2.835,
            confidence=conf / 100.0,
            source="tesseract",
        ))

    return tokens


def get_page_size_mm(input_path: str, dpi: int = 400) -> tuple[float, float]:
    ext = Path(input_path).suffix.lower()
    if ext == ".pdf":
        fitz = _import_pymupdf()
        doc = fitz.open(input_path)
        page = doc[0]
        width_mm = page.rect.width * PT2MM
        height_mm = page.rect.height * PT2MM
        doc.close()
        return width_mm, height_mm

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"読み込み失敗: {input_path}")
    return img.shape[1] * 25.4 / dpi, img.shape[0] * 25.4 / dpi


def load_gray_image(input_path: str, dpi: int = 400) -> np.ndarray:
    ext = Path(input_path).suffix.lower()
    if ext == ".pdf":
        fitz = _import_pymupdf()
        doc = fitz.open(input_path)
        page = doc[0]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
        img_gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height,
            pix.width,
        )
        doc.close()
        return img_gray

    img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"読み込み失敗: {input_path}")
    return img_gray