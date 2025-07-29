import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, use_proto_classifer=False, no_trans=False, 
                 temperature=0.1, dim=128, no_linear=False):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_phase=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = 64 * block.expansion

        self.fc = nn.Linear(self.out_channels, num_classes, bias=False)

        self.use_proto_classifer = use_proto_classifer
        self.temperature = temperature
        
        if self.use_proto_classifer:
            print('Using Proto Classifier, temperature:', self.temperature)

        if no_trans:
            print('Not use trans!')
            self.trans = nn.Identity()
        else:
            if no_linear:
                print('Not linear trans!')
                self.trans = nn.Sequential(
                    nn.Linear(self.out_channels, self.out_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.out_channels, dim),
                )
            else:
                self.trans = nn.Linear(self.out_channels, dim)
        
        self.trans_non = nn.Linear(self.out_channels, dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_feats=False, return_feats_list=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feats0 = x

        x = self.layer1(x)
        feats1 = x

        x = self.layer2(x)
        feats2 = x

        x = self.layer3(x)
        feats3 = x

        x = self.avgpool(x)
        feats = x.view(x.size(0), -1)
        outputs = self.fc(feats)

        con_feats = self.trans(feats)
        
        if return_feats_list:
            return outputs, feats, con_feats, [feats0, feats1, feats2, feats3]
        
        if return_feats:
            return outputs, feats, con_feats, feats3
            # 原型更改
            # return outputs, con_feats, feats, non_feats
        return outputs

def resnet20(num_classes, pretrained=False, progress=True, use_proto_classifer=False, **kwargs):
    n = 3
    model = ResNet(BasicBlock, [n, n, n], use_proto_classifer=use_proto_classifer, **kwargs)
    num_ftrs = model.fc.in_features
    if use_proto_classifer:
        print('Using Proto Classifier')
        model.fc = nn.Linear(num_ftrs, num_classes, bias=False)
    else:
        print('Using Params Classifier')
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model

