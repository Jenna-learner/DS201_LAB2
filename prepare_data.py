# resize trước toàn bộ ảnh (cạnh ngắn 256) cho train nhanh hơn, ảnh gốc khá to mà colab chỉ có 2 cpu
# python prepare_data.py --src VinaFood21 --dst VinaFood21_256
import argparse
import os

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="VinaFood21")
parser.add_argument("--dst", default="VinaFood21_256")
parser.add_argument("--size", type=int, default=256)
args = parser.parse_args()

files = []
for root, _, names in os.walk(args.src):
    for name in names:
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(os.path.join(root, name))
print(len(files), "ảnh")

for path in tqdm(files):
    out = os.path.join(args.dst, os.path.relpath(path, args.src))
    out = os.path.splitext(out)[0] + ".jpg"
    if os.path.exists(out):
        continue
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        print("lỗi:", path)
        continue

    w, h = img.size
    scale = args.size / min(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out, quality=95)
