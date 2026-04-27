import time

import numpy as np

from .models import PreprocessResult
from .preprocess import (
    denoise,
    binarize_otsu,
    binarize_adaptive,
    detect_skew,
    deskew,
    super_resolve,
)
from .quality import evaluate_quality


def preprocess_drawing(
    img: np.ndarray,
    use_adaptive:    bool  = False,
    adaptive_block:  int   = 31,
    adaptive_C:      int   = 10,
    denoise_h:       float = 10.0,
    skew_method:     str   = "hough",
    use_sr:          bool  = False,
    sr_scale:        int   = 2,
    tv_weight:       float = 0.03,
    unsharp_amount:  float = 1.5,
    lap_boost:       float = 0.4,
) -> PreprocessResult:
    """
    Phase 1 前処理パイプラインの実行。

    Args:
        img:            グレースケール入力画像
        use_adaptive:   True → 適応的二値化（照明ムラがある場合）
        adaptive_block: 適応的二値化のウィンドウサイズ（奇数）
        adaptive_C:     適応的二値化の定数オフセット
        denoise_h:      ノイズ除去強度（8〜18 の範囲で調整）
        skew_method:    "hough"（推奨）または "projection"
        use_sr:         True → 超解像を適用
        sr_scale:       超解像の拡大倍率（2 or 4）
        tv_weight:      TV デノイズ強度（0.02〜0.05）
        unsharp_amount: アンシャープマスク強度（1.0〜2.0）
        lap_boost:      Laplacian 強調係数（0〜0.5）
    """
    t0 = time.time()

    quality_in = evaluate_quality(img)

    # Step 1: ノイズ除去
    denoised = denoise(img, h=denoise_h)

    # Step 2: 二値化
    if use_adaptive:
        binary = binarize_adaptive(denoised, block_size=adaptive_block, C=adaptive_C)
    else:
        binary, _ = binarize_otsu(denoised)

    # Step 3: 傾き検出（Hough → projection フォールバック）
    angle = detect_skew(binary, method=skew_method)

    # Step 4: 傾き補正
    deskewed = deskew(binary, angle)

    # Step 5: 超解像（任意）
    sr_image = None
    if use_sr:
        sr_image = super_resolve(
            deskewed,
            scale=sr_scale,
            tv_weight=tv_weight,
            unsharp_amount=unsharp_amount,
            lap_boost=lap_boost,
        )

    output_img = sr_image if sr_image is not None else deskewed
    quality_out = evaluate_quality(output_img, binary=output_img)

    return PreprocessResult(
        original=img,
        denoised=denoised,
        binarized=binary,
        deskewed=deskewed,
        sr_image=sr_image,
        skew_angle=angle,
        quality_in=quality_in,
        quality_out=quality_out,
        elapsed_sec=time.time() - t0,
    )
