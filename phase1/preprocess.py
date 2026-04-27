import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle


# ──────────────────────────────────────────────────────────
# Step 1: ノイズ除去
# ──────────────────────────────────────────────────────────

def denoise(img: np.ndarray, h: float = 10) -> np.ndarray:
    """
    Non-Local Means Denoising（非局所平均法）。

    h: フィルタ強度。大きいほどノイズを除去するが細い寸法線も消える。
       - スキャン図面（通常ノイズ）: h=8〜12
       - 古い図面・コピー品:         h=12〜18
       - デジタル出力（ほぼノイズなし）: h=3〜5

    図面は「細い直線」が命なので h を上げすぎると
    0.05mm 幅の寸法補助線が飽和して消える。
    """
    return cv2.fastNlMeansDenoising(
        img,
        h=h,
        templateWindowSize=7,
        searchWindowSize=21,
    )


# ──────────────────────────────────────────────────────────
# Step 2: 二値化
# ──────────────────────────────────────────────────────────

def binarize_otsu(img: np.ndarray) -> tuple:
    """
    Otsu 法による大域的二値化。

    前提: 図面の輝度ヒストグラムは「白地（200〜255）」と
          「黒線（0〜80）」の双峰分布になる → Otsu が最適閾値を自動選択。
    ガウシアンブラーを前置することでヒストグラムを滑らかにし安定化。

    Returns: (二値画像, 採用閾値)
    """
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    thresh, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, float(thresh)


def binarize_adaptive(
    img: np.ndarray,
    block_size: int = 31,
    C: int = 10,
) -> np.ndarray:
    """
    適応的（局所的）二値化。

    Otsu が失敗するケース:
      - 古い青焼き図面（全体的に青みがかり輝度差が小さい）
      - 照明ムラのあるスキャン
      - 鉛筆書きの薄い図面

    block_size: 局所閾値を計算するウィンドウ（奇数）
                大きいほど広い範囲を参照 → ムラへの対応力 UP
                小さいほど局所的 → 細部は取れるがノイズに敏感
    C: 閾値から引くオフセット。大きいほど「白くする」方向。
    """
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C,
    )


# ──────────────────────────────────────────────────────────
# Step 3: 傾き検出・補正（Deskew）
# ──────────────────────────────────────────────────────────

def detect_skew_hough(binary: np.ndarray) -> float:
    """
    Hough 変換による傾き検出（メイン手法）。

    機械図面は外枠・表題欄・寸法線など水平・垂直な長い直線が多い。
    これを利用して「水平に近い線（±15°）」の角度中央値を傾きとする。

    minLineLength を画像幅の 1/4 にすることで短いノイズ線を除外し、
    外枠や長い寸法線だけを対象にする。
    """
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=binary.shape[1] // 4,
        maxLineGap=20,
    )

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -15 < angle < 15:
                angles.append(angle)

    return float(np.median(angles)) if angles else 0.0


def detect_skew_projection(binary: np.ndarray) -> float:
    """
    水平投影プロファイル法による傾き検出（フォールバック）。

    角度を 0.5° ステップで変えながら、各行の黒画素合計値の
    「分散」が最大になる角度を選ぶ。
    正しい角度では黒線が同一行に揃い行間のコントラストが最大化される。

    計算コストが高いため Hough が失敗した場合のみ呼ぶ。
    """
    best_angle, best_score = 0.0, -1.0
    h, w = binary.shape

    for angle in np.arange(-10.0, 10.5, 0.5):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), borderValue=255)
        projection = np.sum(255 - rotated, axis=1, dtype=np.float64)
        score = float(np.var(projection))
        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def detect_skew(binary: np.ndarray, method: str = "hough") -> float:
    """
    傾き検出のエントリポイント。

    method="hough" の場合: Hough 変換で検出し、結果が微小（< 0.05°）
    または method="projection" の場合は投影法でも検出して大きい方を採用。
    method="projection" の場合: 投影法のみ使用。
    """
    angle_hough = detect_skew_hough(binary)
    if method == "projection" or abs(angle_hough) < 0.05:
        angle_proj = detect_skew_projection(binary)
        if method == "projection" or abs(angle_proj) > abs(angle_hough):
            return angle_proj
    return angle_hough


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    """
    angle 度の傾きを補正する（白背景で境界を埋める）。
    回転中心は画像中央、スケールは 1.0（拡縮なし）。
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────
# Step 4: 超解像（Super-Resolution）
# ──────────────────────────────────────────────────────────

def super_resolve(
    img: np.ndarray,
    scale: int = 2,
    tv_weight: float = 0.03,
    unsharp_amount: float = 1.5,
    lap_boost: float = 0.4,
) -> np.ndarray:
    """
    図面特化の超解像パイプライン（外部モデル不要・ローカル完結）。

    【設計思想】
    深層学習 SR（EDSR/RealESRGAN）は自然画像のテクスチャを
    ハルシネーションで補完する。図面に対してはこれが逆効果：
    実在しない寸法数字や線が「生成」されてしまう危険がある。

    そのため図面では「エッジを忠実に保持しながら解像度を上げる」
    アルゴリズムベース SR が推奨される。

    【4ステップ構成】
    1. Lanczos4 アップスケール
       高品質な補間。Bicubic よりリンギングが少ない。

    2. Total Variation (TV) デノイジング
       エッジ（線）を保持しつつ、補間で生じたぼやけを除去する。
       weight が小さいほどエッジ保持優先（図面は 0.02〜0.05 推奨）。

    3. Unsharp Mask
       ガウシアンブラーとの差分でエッジを強調。
       amount=1.5 は寸法線の可読性を上げつつ過剰シャープを防ぐ。

    4. Laplacian Boost
       2次微分で細かいエッジをさらに強調。
       OCR 前処理として文字のエッジを立てるのに有効。

    Parameters:
        scale:          アップスケール倍率（2 or 4 推奨）
        tv_weight:      TV 正則化強度（小 = エッジ優先）
        unsharp_amount: アンシャープマスク強度（1.0〜2.0）
        lap_boost:      Laplacian 強調係数（0 でスキップ相当）

    実用ガイド:
        - 150dpi スキャン → 300dpi: scale=2, tv_weight=0.03
        - 75dpi スキャン  → 300dpi: scale=4, tv_weight=0.02
        - 300dpi 以上のデジタル PDF: SR 不要
    """
    h, w = img.shape

    # 1. Lanczos4 アップスケール
    upscaled = cv2.resize(
        img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4
    )

    # 2. Total Variation デノイジング（エッジ保持）
    up_f = upscaled.astype(np.float64) / 255.0
    tv_denoised = denoise_tv_chambolle(up_f, weight=tv_weight, channel_axis=None)
    tv_u8 = (tv_denoised * 255).clip(0, 255).astype(np.uint8)

    # 3. Unsharp Mask（エッジ強調）
    blur = cv2.GaussianBlur(tv_u8, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(
        tv_u8, 1.0 + unsharp_amount,
        blur, -unsharp_amount,
        0
    )

    # 4. Laplacian Boost（文字・細線のエッジ強調）
    if lap_boost > 0:
        lap = cv2.Laplacian(sharpened, cv2.CV_64F)
        out = np.clip(
            sharpened.astype(np.float64) - lap_boost * lap,
            0, 255
        ).astype(np.uint8)
    else:
        out = sharpened

    return out
