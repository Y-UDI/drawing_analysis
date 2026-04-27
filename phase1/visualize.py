import os
from pathlib import Path

import cv2
import numpy as np

from .models import PreprocessResult


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * target_h / h), target_h))


def make_comparison(result: PreprocessResult, panel_height: int = 400) -> np.ndarray:
    """全ステップを横並びにした比較画像を生成する。"""
    steps = [
        ("1. Original",  result.original),
        ("2. Denoised",  result.denoised),
        ("3. Binarized", result.binarized),
        ("4. Deskewed",  result.deskewed),
    ]
    if result.sr_image is not None:
        steps.append(("5. Super-Res", result.sr_image))

    panels = []
    for label, img in steps:
        resized = _resize_to_height(img, panel_height)
        panel = cv2.copyMakeBorder(resized, 30, 0, 0, 0,
                                   cv2.BORDER_CONSTANT, value=200)
        cv2.putText(panel, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, 0, 1, cv2.LINE_AA)
        panels.append(panel)

    max_w = max(p.shape[1] for p in panels)
    padded = [
        cv2.copyMakeBorder(p, 0, 0, 0, max_w - p.shape[1],
                           cv2.BORDER_CONSTANT, value=200)
        for p in panels
    ]
    return np.hstack(padded)


def save_all(result: PreprocessResult, output_dir: str, stem: str) -> list:
    """全ステップの画像を保存し、ファイルパスのリストを返す。"""
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    pairs = [
        (f"{stem}_01_original.png",  result.original),
        (f"{stem}_02_denoised.png",  result.denoised),
        (f"{stem}_03_binarized.png", result.binarized),
        (f"{stem}_04_deskewed.png",  result.deskewed),
    ]
    if result.sr_image is not None:
        pairs.append((f"{stem}_05_super_res.png", result.sr_image))

    for fname, img in pairs:
        path = f"{output_dir}/{fname}"
        cv2.imwrite(path, img)
        saved.append(path)

    comparison = make_comparison(result)
    cmp_path = f"{output_dir}/{stem}_comparison.png"
    cv2.imwrite(cmp_path, comparison)
    saved.append(cmp_path)

    return saved


def print_report(result: PreprocessResult) -> None:
    """処理結果サマリをコンソールに表示する。"""
    qi, qo = result.quality_in, result.quality_out
    sr_tag = (f"x{result.sr_image.shape[1] // result.deskewed.shape[1]}"
              if result.sr_image is not None else "なし")

    print(f"\n{'='*50}")
    print(f"  Phase 1 前処理レポート")
    print(f"{'─'*50}")
    print(f"  傾き補正     : {result.skew_angle:+.2f}°")
    print(f"  超解像       : {sr_tag}")
    print(f"{'─'*50}")
    print(f"  {'指標':<14}  {'入力':>10}  {'出力':>10}")
    print(f"  {'─'*36}")
    print(f"  {'解像度':<14}  {qi['resolution']:>10}  {qo['resolution']:>10}")
    print(f"  {'鮮鋭度':<14}  {qi['sharpness']:>10.1f}  {qo['sharpness']:>10.1f}")
    print(f"  {'コントラスト':<12}  {qi['contrast']:>10.1f}  {qo['contrast']:>10.1f}")
    if "black_ratio_pct" in qo:
        print(f"  {'黒画素率 [%]':<13}  {'—':>10}  {qo['black_ratio_pct']:>10.2f}")
    if "snr_estimate" in qo:
        print(f"  {'SNR推定':<14}  {'—':>10}  {qo['snr_estimate']:>10.1f}")
    print(f"{'─'*50}")
    print(f"  処理時間     : {result.elapsed_sec:.2f}s")
    print(f"{'='*50}\n")


def print_output_files(saved: list, output_dir: str) -> None:
    """保存したファイルの一覧をコンソールに表示する。"""
    print(f"[OUTPUT] {output_dir}/")
    for f in saved:
        tag = "← 比較画像" if "comparison" in f else ""
        print(f"         {Path(f).name}  {tag}")
