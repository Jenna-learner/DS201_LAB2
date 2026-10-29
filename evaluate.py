import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from data import get_test_loader
from engine import evaluate, get_metrics
from models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_root", default="VinaFood21")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.run_dir, "info.json")):
        raise SystemExit(f"{args.run_dir} chưa train xong, chưa đánh giá được")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(args.run_dir, "best.pt"), map_location=device, weights_only=False)
    cfg = ckpt["args"]
    classes = ckpt["classes"]

    model = build_model(cfg["model"], len(classes), cfg["img_size"], cfg["pool"], cfg["mode"]).to(device)
    model.load_state_dict(ckpt["model"])

    test_loader, test_classes = get_test_loader(args.data_root, cfg["img_size"], args.batch_size, args.num_workers)
    assert test_classes == classes, "thứ tự lớp ở test khác train"

    loss, y_true, y_pred = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    m = get_metrics(y_true, y_pred)
    print(f"test {len(y_true)} ảnh, model ở epoch {ckpt['epoch']}")
    print(f"accuracy {m['accuracy']:.4f} | precision {m['precision']:.4f} | "
          f"recall {m['recall']:.4f} | macro-F1 {m['f1']:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))

    f1_each = f1_score(y_true, y_pred, average=None, labels=range(len(classes)), zero_division=0)
    pd.DataFrame({"class": classes, "f1": f1_each.round(4)}).to_csv(
        os.path.join(args.run_dir, "per_class_f1.csv"), index=False)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 7})
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(os.path.basename(args.run_dir))
    plt.savefig(os.path.join(args.run_dir, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # mấy cặp lớp bị nhầm nhiều nhất
    wrong = cm.copy()
    np.fill_diagonal(wrong, 0)
    print("nhầm nhiều nhất (thật -> đoán):")
    for k in np.argsort(wrong.ravel())[::-1][:5]:
        i, j = divmod(k, len(classes))
        print(f"  {classes[i]} -> {classes[j]}: {wrong[i, j]}")

    m["test_loss"] = loss
    with open(os.path.join(args.run_dir, "test_metrics.json"), "w") as f:
        json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
