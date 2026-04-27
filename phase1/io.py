from pathlib import Path

import cv2
import numpy as np


def load_image(path: str, dpi: int = 300) -> np.ndarray:
    """
    PNG/JPEG/TIFF: グレースケールで直接読み込み。
    PDF: pymupdf (fitz) で指定 dpi にラスタライズ後グレースケール変換。
    製造業では 300dpi が OCR・検出の実用下限、600dpi で高精度。
    """
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PDF 読み込みには pymupdf が必要です\n"
                "  pip install pymupdf"
            )
        doc = fitz.open(path)
        page = doc[0]  # 先頭ページ（多ページ PDF は別途ループ処理）
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        return img

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"読み込み失敗: {path}")
    return img
