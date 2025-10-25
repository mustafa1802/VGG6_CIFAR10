# test.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

# -----------------------------
# Define the same VGG6 model
# -----------------------------
class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.GELU()]
            else:
                layers += [conv2d, nn.GELU()]
            in_channels = v
    return nn.Sequential(*layers)

def vgg(cfg, num_classes=10, batch_norm=True):
    return VGG(make_layers(cfg, batch_norm=batch_norm), num_classes=num_classes)

# -----------------------------
# Model configuration
# -----------------------------
cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']
model_path = "vgg6_cifar10_model.pth"

# -----------------------------
# CIFAR-10 test data loader
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

test_data = datasets.CIFAR10('./data', train=False,
                             transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=32,
                         shuffle=False, num_workers=4)

# -----------------------------
# Load model and weights
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = vgg(cfg_vgg6, num_classes=10, batch_norm=True).to(device)

# Load weights
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# -----------------------------
# Evaluate on test dataset
# -----------------------------
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

test_accuracy = 100. * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
