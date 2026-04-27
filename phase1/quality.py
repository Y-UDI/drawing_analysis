from typing import Optional

import cv2
import numpy as np


def evaluate_quality(img: np.ndarray, binary: Optional[np.ndarray] = None) -> dict:
    """
    図面品質の数値評価。

    sharpness: ラプラシアン分散。Before/After の比較に使う。
               目安: >500 良好、<100 はぼやけ要注意。

    contrast: 輝度の標準偏差。

    black_ratio: 二値化後の黒画素率 [%]。
               正常な機械図面の目安: 3〜20%。

    snr_estimate: 白領域の輝度平均 / 標準偏差。
               白領域はノイズのみのはずなので SNR の代理指標になる。
               目安: >30 良好。
    """
    lap = cv2.Laplacian(img, cv2.CV_64F)
    result = {
        "resolution": f"{img.shape[1]}x{img.shape[0]}",
        "sharpness":  round(float(lap.var()), 1),
        "contrast":   round(float(img.std()), 1),
    }

    if binary is not None:
        black_ratio = float((binary < 128).mean()) * 100
        white_mask = binary > 200
        if white_mask.sum() > 1000:
            noise_std = float(img[white_mask].std())
            signal_mean = float(img[white_mask].mean())
            snr = signal_mean / (noise_std + 1e-6)
        else:
            snr = 0.0
        result["black_ratio_pct"] = round(black_ratio, 2)
        result["snr_estimate"] = round(snr, 1)

    return result
