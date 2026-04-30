# phase1/pdf2image.py

# %%[markdown]
# ## PyMuPDFを使って、PDFファイルを画像に変換するコードです。

# %%
# import
import fitz  # PyMuPDF
import os
from pathlib import Path    

PATH = os.getcwd().split("phase1")[0]
PDF_PATH = Path(PATH, "drawing", "drawing")
IMG_PATH = Path(PATH, "drawing", "images")
print(PATH)
# %% 
# PDFファイルを開く
"""
PyMuPDFを使用して、PDFファイルを画像に変換するコードです。以下のコードは、指定されたPDFファイルを開き、各ページを画像として保存します。
"""
# PDFファイルを開く
pdf_file = fitz.open(PDF_PATH / "square.pdf")

for i, page in enumerate(pdf_file):
    pix = page.get_pixmap(dpi=300)
    pix.save(IMG_PATH / f"page_{i+1}.png")

print(type(pdf_file))
print(pdf_file)
print(type(pix))
print(pix)
# %%[markdown]
# ## OpenCVで前処理

import cv2
import matplotlib.pyplot as plt
"""
画像を読み込んで表示する
"""
img = cv2.imread(IMG_PATH / "page_1.png")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
# %%[markdown]
# ## 二値化

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
plt.imshow(binary, cmap="gray")
plt.axis("off")

# %%[markdown]
# ## ノイズ除去
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
plt.imshow(clean, cmap="gray")
plt.axis("off")

# %%[markdown]
# ## 直線抽出
edges = cv2.Canny(img, 50, 150)
lines = cv2.HoughLinesP(edges, 1, 3.14/180, 100)
line_img = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

# %%[markdown]
# ## 文字抽出
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 10 and h > 10:  # 小さなノイズを除外
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# %%
fig, ax = plt.subplots(6, 5, sharex=True, sharey=True, figsize=(20,20))
ax = ax.flatten()
for i in range(len(contours)):
    img = cv2.drawContours(
        cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), #描画する画像
        contours,                                    #輪郭を保存したリスト
        i,                                        #リストの何番目を描くか
        (0,255,0),                                   #色の指定
        2
    )                                           #線の太さの指定
    ax[i].imshow(img)
    ax[i].set_title('contour{}'.format(i))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

