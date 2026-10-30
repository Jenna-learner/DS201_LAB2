import torch
import torch.nn as nn


def count_params(model, trainable=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)


# Bài 1
class LeNet5(nn.Module):
    def __init__(self, in_ch=3, img_size=64, num_classes=21, pool="avg"):
        super().__init__()
        Pool = nn.AvgPool2d if pool == "avg" else nn.MaxPool2d
        self.c1 = nn.Conv2d(in_ch, 6, kernel_size=5, padding=2)    # pad 2 như hình, giữ nguyên kích thước
        self.s2 = Pool(2, 2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = Pool(2, 2)

        # tính số chiều sau flatten bằng 1 lần forward giả (64x64 -> 16*14*14 = 3136)
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
def conv_bn(in_ch, out_ch, k, stride=1, padding=0):
    # conv -> BatchNorm -> ReLU, có BN rồi nên conv bỏ bias
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU())


class InceptionBlock(nn.Module):
    # 4 nhánh như hình, số channel ra = f1 + f3 + f5 + fp
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp):
        super().__init__()
        self.b1 = conv_bn(in_ch, f1, 1)
        self.b2 = nn.Sequential(conv_bn(in_ch, f3r, 1), conv_bn(f3r, f3, 3, padding=1))
        self.b3 = nn.Sequential(conv_bn(in_ch, f5r, 1), conv_bn(f5r, f5, 5, padding=2))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), conv_bn(in_ch, fp, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogLeNet(nn.Module):
    # theo hình trong đề: 7x7 Conv -> MaxPool -> 1x1 Conv -> 3x3 Conv -> MaxPool
    # -> 2 Inception -> MaxPool -> 5 Inception -> MaxPool -> 2 Inception -> GAP -> FC
    # số channel lấy theo bản gốc (Szegedy et al. 2014), ảnh 64x64: 64 -> 16 -> 8 -> 4 -> 2 -> GAP
    # mỗi conv có thêm BatchNorm (như GoogLeNet của torchvision) để mạng ~22 lớp train ổn định
    def __init__(self, num_classes=21):
        super().__init__()
        self.b1 = nn.Sequential(
            conv_bn(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.b2 = nn.Sequential(
            conv_bn(64, 64, 1),
            conv_bn(64, 192, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.b3 = nn.Sequential(
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),               # 256
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),             # 480
            nn.MaxPool2d(3, stride=2, padding=1))
        self.b4 = nn.Sequential(
            InceptionBlock(480, 192, 96, 208, 16, 48, 64),              # 512
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),             # 512
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),             # 512
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),             # 528
            InceptionBlock(528, 256, 160, 320, 32, 128, 128),           # 832
            nn.MaxPool2d(3, stride=2, padding=1))
        self.b5 = nn.Sequential(
            InceptionBlock(832, 256, 160, 320, 32, 128, 128),           # 832
            InceptionBlock(832, 384, 192, 384, 48, 128, 128),           # 1024
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return self.fc(x)


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
