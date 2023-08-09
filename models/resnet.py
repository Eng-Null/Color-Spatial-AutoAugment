import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models.resnet import resnet50
#from torchvision.models.resnet import resnet18


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #self.shortcut = nn.Sequential()
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #out = self.linear(out)
        return out

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        #ResNet BasicBlock 18 = [2, 2, 2, 2] 32 = [3, 4, 6, 3]  50 = Bottleneck [3, 4, 6, 3]  101 = [3, 4, 23, 3]
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2])

    def forward(self, x):
        rep = self.resnet(x)
        #rep = F.normalize(rep)
        return rep

class ResNet18Proj(nn.Module):
    def __init__(self, out_dim=128):
        super(ResNet18Proj, self).__init__()
        dim_e = BasicBlock.expansion*512
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

class ResNet18Linear(nn.Module):
    def __init__(self, feature_dim=128):
        super(ResNet18Linear, self).__init__()
        dim_e = BasicBlock.expansion*512
        self.linear = nn.Linear(dim_e, feature_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.encoder = ResNet18Encoder()
        self.projection = ResNet18Proj()

    def forward(self, x):
        out = self.encoder(x)
        feature = out.flatten(start_dim=1)
        out = self.projection(feature)
        #out = self.proj(out)
        return feature, out
    
class ResNet50Encoder(nn.Module):
    def __init__(self):
        super(ResNet50Encoder, self).__init__()
        self.resnet = ResNet(Bottleneck, [3, 4, 6, 3])

    def forward(self, x):
        rep = self.resnet(x)
        return rep

class ResNet50Proj(nn.Module):
    def __init__(self, out_dim=128):
        super(ResNet50Proj, self).__init__()
        dim_e = Bottleneck.expansion*512
        hidden_dim = 4096
        self.mlp = nn.Sequential(
                nn.Linear(dim_e, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
                )
        
    def forward(self, x):
        proj = self.mlp(x)
        proj = F.normalize(proj)
        return proj

class ResNet50Linear(nn.Module):
    def __init__(self, feature_dim=128):
        super(ResNet50Linear, self).__init__()
        dim_e = Bottleneck.expansion*512
        self.linear = nn.Linear(dim_e, feature_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.encoder = ResNet50Encoder()
        self.projection = ResNet50Proj()

    def forward(self, x):
        out = self.encoder(x)
        feature = out.flatten(start_dim=1)
        out = self.projection(feature)
        return feature, out