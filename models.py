"""Models.
  hybrid_effnet / hybrid_resnet50 : PROPOSED  pretrained backbone + extra CNN head
                                   (2 conv blocks + Squeeze-Excitation + avg||max pooling)
  effnet_b0 / resnet50            : transfer-learning baselines (backbone + linear)
  simple_cnn                      : CNN trained from scratch (baseline)
Every model exposes .backbone (pretrained part) and .cam_layer (for Grad-CAM).
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


def conv_bn_relu(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // r, 4), 1), nn.ReLU(inplace=True),
                                nn.Conv2d(max(c // r, 4), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x)


def make_backbone(name, pretrained):
    if name == "efficientnet_b0":
        w = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return tvm.efficientnet_b0(weights=w).features, 1280
    if name == "resnet50":
        w = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        m = tvm.resnet50(weights=w)
        return nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4), 2048
    raise ValueError(name)


class HybridNet(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone, c = make_backbone(backbone, pretrained)
        self.head = nn.Sequential(conv_bn_relu(c, 512), conv_bn_relu(512, 256), SEBlock(256))
        self.avg, self.max = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 256), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(256, num_classes))

    @property
    def cam_layer(self):
        return self.head[-1]

    def forward(self, x):
        f = self.head(self.backbone(x))
        f = torch.cat([self.avg(f).flatten(1), self.max(f).flatten(1)], 1)
        return self.classifier(f)


class TransferNet(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone, c = make_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, num_classes))

    @property
    def cam_layer(self):
        return self.backbone[-1]

    def forward(self, x):
        return self.fc(self.pool(self.backbone(x)).flatten(1))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential()  # nothing pretrained
        chans, blocks, i = [32, 64, 128, 256], [], 3
        for o in chans:
            blocks.append(nn.Sequential(conv_bn_relu(i, o), conv_bn_relu(o, o), nn.MaxPool2d(2)))
            i = o
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    @property
    def cam_layer(self):
        return self.features[-1]

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


MODELS = ["hybrid_effnet", "hybrid_resnet50", "effnet_b0", "resnet50", "simple_cnn"]


def build_model(name, num_classes=7, pretrained=True):
    if name == "hybrid_effnet":
        return HybridNet("efficientnet_b0", num_classes, pretrained)
    if name == "hybrid_resnet50":
        return HybridNet("resnet50", num_classes, pretrained)
    if name == "effnet_b0":
        return TransferNet("efficientnet_b0", num_classes, pretrained)
    if name == "resnet50":
        return TransferNet("resnet50", num_classes, pretrained)
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    raise ValueError(f"unknown model {name}; choose from {MODELS}")
