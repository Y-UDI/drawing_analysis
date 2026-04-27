import json
import os
from pathlib import Path

import cv2

from .models import OCRResult
from .visualize import annotate_image


def print_report(result: OCRResult) -> None:
    title_block = result.title_block
    dimensions = result.dimensions

    print(f"\n{'=' * 55}")
    print(f"  Phase 2-A OCR レポート  [{result.engine}]")
    print(f"{'─' * 55}")
    print(f"  テキスト総数  : {len(result.tokens)}")
    print(f"  寸法値検出数  : {len(dimensions)}")
    print(f"  処理時間      : {result.elapsed_sec:.3f}s")

    if dimensions:
        print("\n  【検出寸法値】")
        by_type = {}
        for dimension in dimensions:
            by_type.setdefault(dimension.dim_type, []).append(dimension)
        for dim_type, items in sorted(by_type.items()):
            values = ", ".join(
                f"{dimension.raw_text}  @({dimension.x:.1f},{dimension.y:.1f})mm"
                for dimension in items
            )
            print(f"    {dim_type:<12}: {values}")

    print("\n  【表題欄】")
    for label, value in [
        ("部品名", title_block.part_name),
        ("材料", title_block.material),
        ("材料番号", title_block.mat_number),
        ("縮尺", title_block.scale),
        ("公差", title_block.tolerance),
        ("図番", title_block.drawing_no),
        ("作成者", title_block.creator),
        ("承認者", title_block.approver),
        ("ステータス", title_block.doc_status),
        ("発行日", title_block.issue_date),
    ]:
        if value:
            print(f"    {label:<10}: {value}")
    print(f"{'=' * 55}\n")


def to_json_dict(result: OCRResult) -> dict:
    return {
        "engine": result.engine,
        "elapsed_sec": round(result.elapsed_sec, 3),
        "input": result.input_path,
        "dimensions": [
            {
                "value": dimension.value,
                "raw": dimension.raw_text,
                "type": dimension.dim_type,
                "position_mm": {
                    "x": round(dimension.x, 2),
                    "y": round(dimension.y, 2),
                },
            }
            for dimension in result.dimensions
        ],
        "title_block": {
            "part_name": result.title_block.part_name,
            "material": result.title_block.material,
            "mat_number": result.title_block.mat_number,
            "scale": result.title_block.scale,
            "tolerance": result.title_block.tolerance,
            "drawing_no": result.title_block.drawing_no,
            "creator": result.title_block.creator,
            "approver": result.title_block.approver,
            "doc_status": result.title_block.doc_status,
            "issue_date": result.title_block.issue_date,
        },
        "all_tokens": [
            {
                "text": token.text,
                "category": token.category,
                "position_mm": {"x": round(token.x, 2), "y": round(token.y, 2)},
                "font_size": round(token.font_size, 1),
                "confidence": round(token.confidence, 3),
                "source": token.source,
            }
            for token in result.tokens
        ],
    }


def save_outputs(result: OCRResult, output_dir: str, write_json: bool = False) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(result.input_path).stem
    saved = []

    annotated = annotate_image(result)
    annotated_path = f"{output_dir}/{stem}_ocr_annotated.png"
    cv2.imwrite(annotated_path, annotated)
    saved.append(annotated_path)

    if write_json:
        json_path = f"{output_dir}/{stem}_ocr.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(to_json_dict(result), handle, ensure_ascii=False, indent=2)
        saved.append(json_path)

    return saved


def print_output_files(saved: list[str]) -> None:
    for path in saved:
        print(f"[OUTPUT] {path}")