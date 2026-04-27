from pathlib import Path

import cv2
import numpy as np

from .models import OCRResult


def _import_pymupdf():
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF が必要です: pip install pymupdf") from exc
    return fitz


COLORS = {
    "dimension": (0, 200, 0),
    "titleblock": (0, 130, 220),
    "note": (200, 100, 0),
    "": (150, 150, 150),
}


def annotate_image(result: OCRResult, dpi: int = 150) -> np.ndarray:
    ext = Path(result.input_path).suffix.lower()
    if ext == ".pdf":
        fitz = _import_pymupdf()
        doc = fitz.open(result.input_path)
        page = doc[0]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix)
        base = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height,
            pix.width,
            3,
        ).copy()
        doc.close()
    else:
        base = cv2.imread(result.input_path)

    if base is None:
        raise FileNotFoundError(f"読み込み失敗: {result.input_path}")

    mm2px = dpi / 25.4
    annotated = base.copy()

    for token in result.tokens:
        color = COLORS.get(token.category, (150, 150, 150))
        x0 = int(token.x0 * mm2px)
        y0 = int(token.y0 * mm2px)
        x1 = int(token.x1 * mm2px)
        y1 = int(token.y1 * mm2px)
        cv2.rectangle(annotated, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), color, 2)
        if token.category == "dimension":
            cv2.putText(
                annotated,
                token.text,
                (x0, max(y0 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    legend = [
        ((0, 200, 0), "寸法値 (dimension)"),
        ((0, 130, 220), "表題欄 (titleblock)"),
        ((200, 100, 0), "注記 (note)"),
    ]
    legend_x, legend_y = 12, 28
    for color, label in legend:
        cv2.rectangle(annotated, (legend_x, legend_y - 14), (legend_x + 22, legend_y + 4), color, -1)
        cv2.putText(
            annotated,
            label,
            (legend_x + 30, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        legend_y += 30

    return annotated