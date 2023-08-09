import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                    bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, dropout, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes:
            self.shortcut = conv1x1(inplanes, planes, stride)
            self.use_conv1x1 = True
        else:
            self.use_conv1x1 = False

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)

        if self.use_conv1x1:
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out += shortcut

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, width, num_classes=10, dropout=0.3):
        super(WideResNet, self).__init__()

        layer = (depth - 4) // 6

        self.inplanes = 16
        self.conv = conv3x3(3, 16)
        self.layer1 = self._make_layer(16*width, layer, dropout)
        self.layer2 = self._make_layer(32*width, layer, dropout, stride=2)
        self.layer3 = self._make_layer(64*width, layer, dropout, stride=2)
        self.bn = nn.BatchNorm2d(64*width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64*width, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, planes, blocks, dropout, stride=1):
        layers = []
        for i in range(blocks):
            layers.append(BasicBlock(self.inplanes, planes, dropout, stride if i == 0 else 1))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        #x = F.avg_pool2d(x, 4)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x

class WideResNetEncoder(nn.Module):
    def __init__(self):
        super(WideResNetEncoder, self).__init__()
        #ResNet BasicBlock 18 = [2, 2, 2, 2] 32 = [3, 4, 6, 3]  50 = Bottleneck [3, 4, 6, 3]  101 = [3, 4, 23, 3]
        self.resnet = WideResNet(28, 10, 10)

    def forward(self, x):
        rep = self.resnet(x)
        #rep = F.normalize(rep)
        return rep

class WideResNetProj(nn.Module):
    def __init__(self, out_dim=128):
        super(WideResNetProj, self).__init__()
        dim_e = 640
        #the hidden_dim used in In the original SimCLR paper by Chen et al
        #a value between 2048 and 4096
        hidden_dim = 2048
        self.mlp = nn.Sequential(
                nn.Linear(dim_e, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
                )
        
    def forward(self, x):
        proj = self.mlp(x)
        proj = F.normalize(proj)
        return proj
    
class WideResNet2810(nn.Module):
    def __init__(self):
        super(WideResNet2810, self).__init__()
        self.encoder = WideResNetEncoder()
        self.projection = WideResNetProj()

    def forward(self, x):
        out = self.encoder(x)
        feature = out.flatten(start_dim=1)
        out = self.projection(feature)
        #out = self.proj(out)
        return feature, out
    
class WideResNet2810Linear(nn.Module):
    def __init__(self, feature_dim=128):
        super(WideResNet2810Linear, self).__init__()
        dim_e = 640
        self.linear = nn.Linear(dim_e, feature_dim)

    def forward(self, x):
        out = self.linear(x)
        return out