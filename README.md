# DS201 - Lab 2: CNN

Dataset: VinaFood21 (21 lớp món ăn Việt Nam)

- Bài 1: LeNet-5, chạy với AvgPool rồi đổi sang MaxPool
- Bài 2: GoogLeNet rút gọn (4 Inception block, GAP)
- Bài 3: ResNet-18 train từ đầu, có và không có augmentation
- Bài 4: ResNet-50 pretrained (`microsoft/resnet-50`), feature extraction và fine-tuning

Kết quả và nhận xét ở `Lab2_23521117.ipynb`, bảng tổng hợp ở `results.md`.

## Các file

- `data.py`: transform, chia train/val (10% train, seed 42, lưu ở `splits/` để 4 bài dùng chung), DataLoader
- `models.py`: LeNet5, GoogLeNet, ResNet18, PretrainedResnet
- `engine.py`: train 1 epoch, evaluate, tính accuracy/precision/recall/macro-F1
- `train.py`: train 1 model, lưu `best.pt` theo macro-F1 trên val, `history.csv`, `loss.png`
- `evaluate.py`: đánh giá trên test, F1 từng lớp, confusion matrix
- `summary.py`: gom kết quả các bài thành `results.md`
- `prepare_data.py`: resize ảnh trước cho train nhanh hơn
- `practice/`: notebook thực hành trên lớp

## Chạy

```
pip install -r requirements.txt

python train.py --model lenet --pool avg
python train.py --model lenet --pool max
python train.py --model googlenet --amp
python train.py --model resnet18 --amp
python train.py --model resnet18 --no_aug --amp
python train.py --model resnet50 --mode fe --amp
python train.py --model resnet50 --mode ft --amp

python evaluate.py --run_dir runs/lenet_avg
...
python summary.py
```

Mặc định: lenet, googlenet ảnh 64, 20 epoch; resnet18 ảnh 112, 30 epoch; resnet50 ảnh 224, 10 epoch.
Adam lr 1e-3, riêng resnet50 fine-tuning lr 1e-4, batch 32.

Mình chạy trên Colab T4. Thư mục `runs/` với `splits/` để trên Google Drive nên bị ngắt thì chạy lại là
train tiếp được (sau mỗi epoch có lưu `last.pt`), run nào xong rồi thì tự bỏ qua.
