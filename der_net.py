import copy
import logging
import torch
from torch import nn
from cifar_resnet import resnet32, resnet20
from cifar_resnet_t import resnet18
from torch.nn import functional as F

def get_convnet(args, pretrained=False):
    name = args.model.lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained,args=args)
    elif name == "resnet20":
        return resnet20(pretrained=pretrained,args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(name))

class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}


class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.convnet_type = args.model
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.trans_dim = args.dim
        self.fc = None
        self.trans = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args
        self.old_classify_weight = None
        self.old_classify_bias = None

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logics: self.fc(features)}
        feats = self.trans(features)["logits"]

        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]  # 模型的输出
        old_logits = None
        if self.old_classify_weight is not None:
            old_logits = F.linear(features[:, :(features.shape[1] - self.out_dim)], self.old_classify_weight.cuda(features.device), self.old_classify_bias.cuda(features.device))

        out.update({"aux_logits": aux_logits, "features": features, "old_logits": old_logits, "con_feats": feats})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args))
        else:
            self.convnets.append(get_convnet(self.args))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict()) # 把上一个模型的参数复制到新的任务中

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:                   # 把上一个模型的分类器也扩展到新的分类器上，原来的旧分类器删掉
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

            self.old_classify_weight = weight
            self.old_classify_bias = bias
        
        del self.fc
        self.fc = fc

        trans = self.generate_fc(self.feature_dim, self.trans_dim)
        if self.trans is not None:
            nb_output = self.trans.in_features
            weight = copy.deepcopy(self.trans.weight.data)
            bias = copy.deepcopy(self.trans.bias.data)
            trans.weight.data[:, :nb_output] = weight
            trans.bias.data[:] = bias

            self.old_trans_weight = weight
            self.old_trans_bias = bias

        del self.trans
        self.trans = trans

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)   # 每一个任务单独的分类器都是重新初始化的

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew  # 在半监督的设置下，这个值通常是小于1的，也就是新的norm会过大
        logging.info(f"logging alignweights,gamma = {gamma}")
        self.fc.weight.data[-increment:, :] *= gamma