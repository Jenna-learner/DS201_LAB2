import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUNS = [
    ("lenet_avg", "LeNet-5 (AvgPool)"),
    ("lenet_max", "LeNet-5 (MaxPool)"),
    ("googlenet", "GoogLeNet"),
    ("resnet18_aug", "ResNet-18 từ đầu (có aug)"),
    ("resnet18_noaug", "ResNet-18 từ đầu (không aug)"),
    ("resnet50_fe", "ResNet-50 pretrained (feature extraction)"),
    ("resnet50_ft", "ResNet-50 pretrained (fine-tuning)"),
]


def load(name):
    with open(f"runs/{name}/info.json") as f:
        info = json.load(f)
    with open(f"runs/{name}/test_metrics.json") as f:
        m = json.load(f)
    return info, m


done = [(name, label) for name, label in RUNS if os.path.exists(f"runs/{name}/test_metrics.json")]
if not done:
    raise SystemExit("chưa có run nào đánh giá xong")
results = {name: load(name) for name, _ in done}

lines = ["## Bảng tổng hợp (tập test)", "",
         "| Mô hình | Resize | Tham số | Tham số học | Giây/epoch | Accuracy | Precision | Recall | macro-F1 |",
         "|---|---|---|---|---|---|---|---|---|"]
for name, label in done:
    info, m = results[name]
    lines.append(f"| {label} | {info['img_size']}x{info['img_size']} | {info['n_params']:,} | "
                 f"{info['n_trainable']:,} | {info['sec_per_epoch']:.1f} | {m['accuracy']:.4f} | "
                 f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")

lines += ["", "## So sánh macro-F1", ""]
pairs = [("lenet_max", "lenet_avg"), ("googlenet", "lenet_avg"), ("resnet18_aug", "resnet18_noaug"),
         ("resnet50_ft", "resnet50_fe"), ("resnet50_ft", "resnet18_aug")]
for a, b in pairs:
    if a in results and b in results:
        fa, fb = results[a][1]["f1"], results[b][1]["f1"]
        lines.append(f"- {a} vs {b}: {fa:.4f} vs {fb:.4f} (chênh {fa - fb:+.4f})")

# epoch val loss thấp nhất = lúc 2 đường loss bắt đầu tách nhau
lines += ["", "## Epoch val loss thấp nhất", "", "| Run | Epoch | val loss min | val loss cuối |", "|---|---|---|---|"]
for name, _ in done:
    h = pd.read_csv(f"runs/{name}/history.csv")
    best = h.loc[h.val_loss.idxmin()]
    lines.append(f"| {name} | {int(best.epoch)} | {best.val_loss:.4f} | {h.val_loss.iloc[-1]:.4f} |")

f1_table = pd.concat([pd.read_csv(f"runs/{name}/per_class_f1.csv", index_col="class")["f1"].rename(name)
                      for name, _ in done], axis=1)
lines += ["", "## F1 từng lớp", "", "| Lớp | " + " | ".join(f1_table.columns) + " |",
          "|---|" + "---|" * len(f1_table.columns)]
for cls, row in f1_table.iterrows():
    lines.append(f"| {cls} | " + " | ".join(f"{v:.3f}" for v in row) + " |")

with open("results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

# Bài 3: vẽ chung loss của 2 lần chạy resnet18
if "resnet18_aug" in results and "resnet18_noaug" in results:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, ["resnet18_aug", "resnet18_noaug"]):
        h = pd.read_csv(f"runs/{name}/history.csv")
        best = int(h.loc[h.val_loss.idxmin(), "epoch"])
        ax.plot(h.epoch, h.train_loss, "o-", ms=3, label="train loss")
        ax.plot(h.epoch, h.val_loss, "o-", ms=3, label="val loss")
        ax.axvline(best, color="gray", ls="--")
        ax.set_title(f"{name} (val loss min ở epoch {best})")
        ax.set_xlabel("epoch")
        ax.legend()
    axes[0].set_ylabel("loss")
    plt.savefig("runs/resnet18_compare.png", dpi=120, bbox_inches="tight")
    plt.close()
