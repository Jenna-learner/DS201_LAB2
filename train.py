import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import get_train_val_loaders
from engine import evaluate, get_metrics, train_one_epoch
from models import build_model, count_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["lenet", "googlenet", "resnet18", "resnet50"])
    parser.add_argument("--data_root", default="VinaFood21")
    parser.add_argument("--pool", default="avg", help="lenet: avg hoặc max")
    parser.add_argument("--mode", default="ft", help="resnet50: fe (feature extraction) hoặc ft (fine-tuning)")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--run_name")
    parser.add_argument("--out_dir", default="runs")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    # cấu hình theo đề
    if args.img_size is None:
        args.img_size = {"lenet": 64, "googlenet": 64, "resnet18": 112, "resnet50": 224}[args.model]
    if args.epochs is None:
        args.epochs = {"lenet": 20, "googlenet": 20, "resnet18": 30, "resnet50": 10}[args.model]
    if args.lr is None:
        args.lr = 1e-4 if args.model == "resnet50" and args.mode == "ft" else 1e-3

    if args.run_name is None:
        if args.model == "lenet":
            args.run_name = "lenet_" + args.pool
        elif args.model == "resnet18":
            args.run_name = "resnet18_noaug" if args.no_aug else "resnet18_aug"
        elif args.model == "resnet50":
            args.run_name = "resnet50_" + args.mode
        else:
            args.run_name = args.model
    return args


def plot_loss(history, path, title):
    epochs = [h["epoch"] for h in history]
    best = min(history, key=lambda h: h["val_loss"])["epoch"]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [h["train_loss"] for h in history], "o-", ms=3, label="train loss")
    plt.plot(epochs, [h["val_loss"] for h in history], "o-", ms=3, label="val loss")
    plt.axvline(best, color="gray", ls="--", label=f"val loss min (epoch {best})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    save_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    if args.retrain:
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))
    if os.path.exists(os.path.join(save_dir, "info.json")):
        print(args.run_name, "đã train xong, bỏ qua (thêm --retrain nếu muốn train lại)")
        return

    train_loader, val_loader, classes = get_train_val_loaders(
        args.data_root, args.img_size, args.batch_size, not args.no_aug, args.num_workers, args.seed)

    model = build_model(args.model, len(classes), args.img_size, args.pool, args.mode).to(device)
    n_params = count_params(model)
    n_trainable = count_params(model, trainable=True)
    print(f"{args.run_name} | {device} | train {len(train_loader.dataset)} ảnh | val {len(val_loader.dataset)} ảnh")
    print(f"params {n_params:,} (trainable {n_trainable:,}) | img {args.img_size} | lr {args.lr} | aug {not args.no_aug}")

    criterion = nn.CrossEntropyLoss()
    # chỉ đưa tham số cần học vào optimizer (resnet50 fe chỉ học classifier)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = []
    best_f1 = -1
    start_epoch = 1
    last_path = os.path.join(save_dir, "last.pt")
    if os.path.exists(last_path):
        # colab bị ngắt giữa chừng thì train tiếp từ epoch đang dở
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        history = ckpt["history"]
        best_f1 = ckpt["best_f1"]
        start_epoch = ckpt["epoch"] + 1
        print("train tiếp từ epoch", start_epoch)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        sec = time.time() - t0
        val_loss, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        val = get_metrics(y_true, y_pred)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val["accuracy"], "val_f1": val["f1"], "sec": sec})
        print(f"epoch {epoch}/{args.epochs} - train loss {train_loss:.4f} acc {train_acc:.4f} - "
              f"val loss {val_loss:.4f} acc {val['accuracy']:.4f} f1 {val['f1']:.4f} - {sec:.0f}s")

        # giữ model có macro-F1 trên val cao nhất
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({"model": model.state_dict(), "classes": classes, "args": vars(args), "epoch": epoch},
                       os.path.join(save_dir, "best.pt"))

        # lưu lại sau mỗi epoch để còn train tiếp, ghi ra file tạm trước cho khỏi hỏng file nếu bị ngắt lúc đang lưu
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "history": history, "best_f1": best_f1, "epoch": epoch}, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)

    with open(os.path.join(save_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    plot_loss(history, os.path.join(save_dir, "loss.png"), args.run_name)

    # bỏ epoch đầu khi tính thời gian vì epoch đầu chậm hơn bình thường
    secs = [h["sec"] for h in history]
    info = vars(args).copy()
    info.update({
        "n_classes": len(classes),
        "n_params": n_params,
        "n_trainable": n_trainable,
        "sec_per_epoch": float(np.mean(secs[1:])) if len(secs) > 1 else secs[0],
        "best_epoch": max(history, key=lambda h: h["val_f1"])["epoch"],
        "best_val_f1": best_f1,
    })
    with open(os.path.join(save_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    os.remove(last_path)
    print(f"xong, best val f1 {best_f1:.4f} ở epoch {info['best_epoch']}, ~{info['sec_per_epoch']:.0f}s/epoch")


if __name__ == "__main__":
    main()
