"""
図面解析 Phase 1 — CLI エントリポイント

Usage:
    # 基本
    python phase1.py drawing.png

    # PDF（300dpi でラスタライズ）
    python phase1.py drawing.pdf --dpi 300

    # 低解像度スキャンに超解像を適用
    python phase1.py scan_150dpi.png --sr --sr_scale 2

    # 照明ムラが強い図面（適応的二値化）
    python phase1.py old_scan.png --adaptive

    # 出力先指定
    python phase1.py drawing.png --output_dir ./results

Dependencies:
    pip install opencv-contrib-python scikit-image scipy
    pip install pymupdf  # PDF対応（任意）
"""

import argparse
from pathlib import Path

from .io import load_image
from .pipeline import preprocess_drawing
from .visualize import print_report, save_all, print_output_files


def _REMOVE_load_image(path: str, dpi: int = 300):
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


# ──────────────────────────────────────────────────────────
# 品質評価指標
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
# メインパイプライン
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
# 可視化・出力ユーティリティ
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="図面解析 Phase 1 — 前処理パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python drawing_preprocess.py drawing.png
  python drawing_preprocess.py drawing.pdf --dpi 300
  python drawing_preprocess.py scan.png --sr --sr_scale 2
  python drawing_preprocess.py old.png --adaptive --denoise_h 15
  python drawing_preprocess.py drawing.png --skew_method projection
""",
    )
    p.add_argument("input")
    p.add_argument("--dpi",            type=int,   default=300)
    p.add_argument("--output_dir",     default="./output")
    p.add_argument("--adaptive",       action="store_true")
    p.add_argument("--adaptive_block", type=int,   default=31)
    p.add_argument("--adaptive_C",     type=int,   default=10)
    p.add_argument("--denoise_h",      type=float, default=10.0)
    p.add_argument("--skew_method",    choices=["hough", "projection"], default="hough")
    p.add_argument("--sr",             action="store_true")
    p.add_argument("--sr_scale",       type=int,   default=2, choices=[2, 4])
    p.add_argument("--tv_weight",      type=float, default=0.03)
    p.add_argument("--unsharp",        type=float, default=1.5)
    p.add_argument("--lap_boost",      type=float, default=0.4)
    return p


def print_output_files(saved: list, output_dir: str) -> None:
    """保存したファイルの一覧をコンソールに表示する。"""
    print(f"[OUTPUT] {output_dir}/")
    for f in saved:
        tag = "← 比較画像" if "comparison" in f else ""
        print(f"         {Path(f).name}  {tag}")


def main():
    args = build_parser().parse_args()
    print(f"[INPUT]  {args.input}")

    img = load_image(args.input, dpi=args.dpi)
    print(f"[LOAD]   {img.shape[1]}x{img.shape[0]} px")

    result = preprocess_drawing(
        img,
        use_adaptive=args.adaptive,
        adaptive_block=args.adaptive_block,
        adaptive_C=args.adaptive_C,
        denoise_h=args.denoise_h,
        skew_method=args.skew_method,
        use_sr=args.sr,
        sr_scale=args.sr_scale,
        tv_weight=args.tv_weight,
        unsharp_amount=args.unsharp,
        lap_boost=args.lap_boost,
    )

    print_report(result)

    stem = Path(args.input).stem
    saved = save_all(result, args.output_dir, stem)

    print_output_files(saved, args.output_dir)


if __name__ == "__main__":
    main()