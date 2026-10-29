import torch
import torch.nn as nn


def count_params(model, trainable=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)


# Bài 1
class LeNet5(nn.Module):
    def __init__(self, in_ch=3, img_size=64, num_classes=21, pool="avg"):
        super().__init__()
        Pool = nn.AvgPool2d if pool == "avg" else nn.MaxPool2d
        self.c1 = nn.Conv2d(in_ch, 6, kernel_size=5)
        self.s2 = Pool(2, 2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = Pool(2, 2)

        # tính số chiều sau flatten bằng 1 lần forward giả (64x64 -> 16*13*13 = 2704)
        with torch.no_grad():
            x = torch.zeros(1, in_ch, img_size, img_size)
            n_flat = self.s4(self.c3(self.s2(self.c1(x)))).numel()

        self.f5 = nn.Linear(n_flat, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.s2(torch.relu(self.c1(x)))
        x = self.s4(torch.relu(self.c3(x)))
        x = x.flatten(1)
        x = torch.relu(self.f5(x))
        x = torch.relu(self.f6(x))
        return self.out(x)


# Bài 2
class InceptionBlock(nn.Module):
    # số channel ra = f1 + f3 + f5 + fp
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, f1, 1), nn.ReLU())
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, f3r, 1), nn.ReLU(),
            nn.Conv2d(f3r, f3, 3, padding=1), nn.ReLU())
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, f5r, 1), nn.ReLU(),
            nn.Conv2d(f5r, f5, 5, padding=2), nn.ReLU())
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, fp, 1), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogLeNet(nn.Module):
    # bản rút gọn 4 block, số channel lấy khoảng 1/2 inception 3a, 3b, 4a, 4b của bản gốc
    # ảnh 64x64 nên stem chỉ giảm 4 lần (64 -> 16)
    def __init__(self, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),                       # 32x32
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 96, 3, padding=1, bias=False), nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),                       # 16x16
        )
        self.inc3a = InceptionBlock(96, 32, 48, 64, 8, 16, 16)          # 128
        self.inc3b = InceptionBlock(128, 64, 64, 96, 16, 48, 32)        # 240
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)                # 8x8
        self.inc4a = InceptionBlock(240, 96, 48, 104, 8, 24, 32)        # 256
        self.inc4b = InceptionBlock(256, 80, 56, 112, 12, 32, 32)       # 256
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inc3b(self.inc3a(x))
        x = self.pool(x)
        x = self.inc4b(self.inc4a(x))
        x = self.gap(x).flatten(1)
        return self.fc(self.dropout(x))


# Bài 3
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = self.make_layer(64, 64, stride=1)
        self.layer2 = self.make_layer(64, 128, stride=2)
        self.layer3 = self.make_layer(128, 256, stride=2)
        self.layer4 = self.make_layer(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_ch, out_ch, stride):
        # block đầu đổi channel + giảm kích thước, block sau giữ nguyên
        return nn.Sequential(BasicBlock(in_ch, out_ch, stride), BasicBlock(out_ch, out_ch, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.pool(x).flatten(1))


# Bài 4
class PretrainedResnet(nn.Module):
    def __init__(self, num_classes=21, freeze_backbone=False):
        super().__init__()
        from transformers import ResNetForImageClassification
        base = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.resnet = base.resnet
        self.classifier = nn.Linear(2048, num_classes)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # đóng băng backbone thì BN cũng giữ nguyên thống kê của ImageNet
        if self.freeze_backbone:
            self.resnet.eval()
        return self

    def forward(self, x):
        features = self.resnet(x).pooler_output     # [B, 2048, 1, 1]
        return self.classifier(features.flatten(1))


def build_model(name, num_classes=21, img_size=64, pool="avg", mode="ft"):
    if name == "lenet":
        return LeNet5(3, img_size, num_classes, pool)
    if name == "googlenet":
        return GoogLeNet(num_classes)
    if name == "resnet18":
        return ResNet18(num_classes)
    if name == "resnet50":
        return PretrainedResnet(num_classes, freeze_backbone=(mode == "fe"))
    raise ValueError(name)


if __name__ == "__main__":
    for name, size in [("lenet", 64), ("googlenet", 64), ("resnet18", 112)]:
        model = build_model(name, 21, size)
        out = model(torch.rand(2, 3, size, size))
        print(name, tuple(out.shape), f"{count_params(model):,}")
