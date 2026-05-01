# phase2/embedding.py

# %%[markdown]
# ## CLIPを使って画像をembeddingする
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt 

PATH = os.getcwd().split("phase")[0]
PDF_PATH = Path(PATH, "drawing", "drawing")
IMG_PATH = Path(PATH, "drawing", "images")
print(PATH)
# %%
"""
モデルのロード
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()
# %%
"""
画像の読み込み
"""
image = Image.open(IMG_PATH / "page_1.png").convert("RGB")
plt.imshow(image)
plt.axis("off")
# %%
inputs = processor(
    images=image,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    image_outputs = model.get_image_features(**inputs)
    image_features = image_outputs.pooler_output if hasattr(image_outputs, "pooler_output") else image_outputs
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

embedding = image_features.cpu().numpy()

print(embedding.shape)  # (1, 512)
print(embedding)
# %%

