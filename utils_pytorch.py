#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function, division
import math
import torch
import torchvision
from pathlib import Path
from torchvision.datasets import ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.init as init
import os
import gc
import os.path as osp

import subprocess
import pickle
import numpy as np
import random


def get_data_file(filename, data_dir, label2id, unlabel=False):
    data = []
    targets = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(os.path.join(data_dir, line.strip()))
            targets.append(label2id[line.strip().split("/")[1]])
    if unlabel:
        return np.array(data)

    return np.array(data), np.array(targets)


def get_data_file_unlabeled(filename, data_dir, label2id, unlabel=False):
    data = []
    targets = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(os.path.join(data_dir, line.strip()))
            targets.append(label2id[line.strip().split("/")[1]])
    if unlabel:
        return np.array(data)

    return np.array(data), np.array(targets)


def get_data_file_cifar(data_dir, base_session, index, train, unlabel=False, class_list=None, unlabels_num=None, return_ulb=False, labels_num=None, dataset='cifar100', random=True):

    def SelectfromDefault(data, targets, index, num_per_class=None, return_ulb=False):
        data_tmp = []
        targets_tmp = []
        udata_tmp = []
        utargets_tmp = []
        
        for i in index:
            ind_cl = np.where(targets == i)[0]
            if num_per_class is not None:
                if len(data_tmp) == 0:
                    data_tmp = data[ind_cl][:num_per_class]
                    targets_tmp = targets[ind_cl][:num_per_class]
                    udata_tmp = data[ind_cl][num_per_class:]
                    utargets_tmp = targets[ind_cl][num_per_class:]
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl][:num_per_class]))
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl][:num_per_class]))
                    udata_tmp = np.vstack((udata_tmp, data[ind_cl][num_per_class:]))
                    utargets_tmp = np.hstack((utargets_tmp, targets[ind_cl][num_per_class:]))
            else:
                if len(data_tmp) == 0:
                    data_tmp = data[ind_cl]
                    targets_tmp = targets[ind_cl]
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl]))
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))
        
        if return_ulb:
            return data_tmp, targets_tmp, udata_tmp, utargets_tmp
        
        return data_tmp, targets_tmp

    def NewClassSelector(data, targets, index, num_per_class=None):
        data_tmp = []
        targets_tmp = []
        ind_list = [int(i) for i in index]
        ind_np = np.array(ind_list, dtype=int)
        
        if len(ind_np) == 25:
            index = ind_np.reshape((5,5))
            for i in index:
                ind_cl = i
                if len(data_tmp) == 0:
                    data_tmp = data[ind_cl]
                    targets_tmp = targets[ind_cl]
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl]))
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))
        else:
            data_tmp, targets_tmp = data[ind_np], targets[ind_np]
        
        return data_tmp, targets_tmp

    def NewClassSelectorForUnlabels(data, targets, index, class_list, num_per_class=None):
        # 确保 data 和 targets 是 NumPy 数组
        data = np.array(data)
        targets = np.array(targets)
        
        # 使用 NumPy 数组操作来提高性能
        all_index = np.concatenate([np.where(targets == i)[0] for i in class_list])
        
        ind_np = np.array([int(i) for i in index])
        
        unlabels_index = np.setdiff1d(all_index, ind_np)
        
        unlabels_data, unlabels_targets = data[unlabels_index], targets[unlabels_index]

        if num_per_class is not None:
            for i in class_list:
                ind_cl = np.where(unlabels_targets == i)[0]
                if len(ind_cl) > num_per_class:
                    ind_cl = np.random.choice(ind_cl, num_per_class, replace=False)
                if len(data_tmp) == 0:
                    data_tmp = unlabels_data[ind_cl]
                    targets_tmp = unlabels_targets[ind_cl]
                else:
                    data_tmp = np.vstack((data_tmp, unlabels_data[ind_cl]))
                    targets_tmp = np.hstack((targets_tmp, unlabels_targets[ind_cl]))
        else:
            data_tmp, targets_tmp = unlabels_data, unlabels_targets

        return data_tmp, targets_tmp
    
    def NewClassSelectorForLabelsAndUnlabels(data, targets, index, class_list, num_per_class=None):
        # 确保 data 和 targets 是 NumPy 数组
        data = np.array(data)
        targets = np.array(targets)
        
        ind_np = np.array([int(i) for i in index])

        all_labels_data, all_labels_targets = data[ind_np], targets[ind_np]

        all_index = np.concatenate([np.where(targets == i)[0] for i in class_list])
        labels_index = np.concatenate([ind_np[all_labels_targets == i] for i in class_list])

        unlabels_index = np.setdiff1d(all_index, labels_index)
        
        unlabels_data, unlabels_targets = data[unlabels_index], targets[unlabels_index]
        labels_data, labels_targets = data[labels_index], targets[labels_index]

        return labels_data, labels_targets, unlabels_data, unlabels_targets
        

    if dataset == 'cifar100':
        print('==> Preparing cifar100 data..')
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
    elif dataset == 'cifar10':
        print('==> Preparing cifar10 data..')
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    else:
        raise ValueError('dataset must be cifar10 or cifar100')
    
    if return_ulb:
        if random:
            data, targets, u_data, u_targets = SelectfromDefault(trainset.data, np.array(trainset.targets), index, num_per_class=labels_num, return_ulb=return_ulb)
        else:
            data, targets, u_data, u_targets = NewClassSelectorForLabelsAndUnlabels(trainset.data, np.array(trainset.targets), index, class_list, num_per_class=labels_num)
        return data, targets, u_data, u_targets

    if unlabel:
        if unlabels_num is not None:
            num_per_class = unlabels_num // len(class_list) + 1
        else:
            num_per_class = None
        data, targets = NewClassSelectorForUnlabels(trainset.data, np.array(trainset.targets), index, class_list, num_per_class)
    else:
        if train:
            if base_session:
                data, targets = SelectfromDefault(trainset.data, np.array(trainset.targets), index)
            else:
                data, targets = NewClassSelector(trainset.data, np.array(trainset.targets), index)
        else:
            # data, targets = SelectfromDefault(testset.data, np.array(testset.targets), index)
            if base_session:
                data, targets = SelectfromDefault(testset.data, np.array(testset.targets), index)
            else:
                data, targets = NewClassSelector(testset.data, np.array(testset.targets), index)

    assert len(data) == len(targets)
    return data, targets


def get_data_file_miniimagenet(root, base_session, index, train, unlabel=False, class_list=None, unlabels_num=None, return_ulb=False, labels_num=None, dataset='miniimagenet', index_path=None):
    np.random.seed(1993)
    
    root = os.path.expanduser(root)
    
    IMAGE_PATH = os.path.join(root, 'miniimagenet/images')
    SPLIT_PATH = os.path.join(root, 'miniimagenet/split')

    def SelectfromTxt(data2label, index_path):
        #select from txt file, and make cooresponding mampping.
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(data, targets, index, num_per_class=None, return_ulb=False):
        #select from csv file, choose all instances from this class.
        data_tmp = []
        targets_tmp = []
        udata_tmp = []
        utargets_tmp = []

        if num_per_class is not None:
            for i in index:
                num_tmp = 0
                ind_cl = np.where(i == targets)[0]
                for j in ind_cl:
                    if num_tmp < num_per_class:
                        data_tmp.append(data[j])
                        targets_tmp.append(targets[j])
                    else:
                        udata_tmp.append(data[j])
                        utargets_tmp.append(targets[j])
                    num_tmp += 1
        else:
            for i in index:
                ind_cl = np.where(i == targets)[0]
                for j in ind_cl:
                    data_tmp.append(data[j])
                    targets_tmp.append(targets[j])
        
        if return_ulb:
            return data_tmp, targets_tmp, udata_tmp, utargets_tmp
        
        return data_tmp, targets_tmp
    
    def SelectfromTxtAndClasses(data, targets, index, data2label, index_path):
        data_all = []
        targets_all = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_all.append(data[j])
                targets_all.append(targets[j])
        # index=[]
        # lines = [x.strip() for x in open(index_path, 'r').readlines()]
        # for line in lines:
        #     index.append(line.split('/')[3])
        # data_tmp = []
        # targets_tmp = []
        # for i in index:
        #     img_path = os.path.join(IMAGE_PATH, i)
        #     data_tmp.append(img_path)
        #     targets_tmp.append(data2label[img_path])
        
        return data_all, targets_all
    
    if train:
        setname = 'train'
    else:
        setname = 'test'

    csv_path = osp.join(SPLIT_PATH, setname + '.csv')
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

    data = []
    targets = []
    data2label = {}
    lb = -1

    wnids = []

    for l in lines:
        name, wnid = l.split(',')
        path = osp.join(IMAGE_PATH, name)
        if wnid not in wnids:
            wnids.append(wnid)
            lb += 1
        data.append(path)
        targets.append(lb)
        data2label[path] = lb
    
    if return_ulb:
        data, targets, u_data, u_targets = SelectfromClasses(data, targets, index, num_per_class=labels_num, return_ulb=return_ulb)
        return data, targets, u_data, u_targets

    if train:
        if unlabel:
            select_data, select_targets = SelectfromTxtAndClasses(data, targets, class_list, data2label, index)
        else:
            if base_session:
                select_data, select_targets = SelectfromClasses(data, targets, index)
            else:
                select_data, select_targets = SelectfromTxt(data2label, index)
    else:
        # select_data, select_targets = SelectfromClasses(data, targets, index)
        if base_session:
            select_data, select_targets = SelectfromClasses(data, targets, index)
        else:
            select_data, select_targets = SelectfromTxt(data2label, index)

    assert len(select_data) == len(select_data)
    return select_data, select_targets


def get_data_file_imagenet100(root, base_session, index, train, unlabel=False, class_list=None, unlabels_num=None, return_ulb=False, percentage=None, dataset='imagenet100', index_path=None):
    
    def SelectfromClasses(data, targets, index, num_per_class=None, return_ulb=False):
        #select from csv file, choose all instances from this class.
        data_tmp = []
        targets_tmp = []
        udata_tmp = []
        utargets_tmp = []

        if num_per_class is not None:
            for i in index:
                num_tmp = 0
                ind_cl = np.where(i == targets)[0]
                for j in ind_cl:
                    if num_tmp < num_per_class:
                        data_tmp.append(data[j])
                        targets_tmp.append(targets[j])
                    else:
                        udata_tmp.append(data[j])
                        utargets_tmp.append(targets[j])
                    num_tmp += 1
        else:
            for i in index:
                ind_cl = np.where(i == targets)[0]
                for j in ind_cl:
                    data_tmp.append(data[j])
                    targets_tmp.append(targets[j])
        
        if return_ulb:
            return data_tmp, targets_tmp, udata_tmp, utargets_tmp
        
        return data_tmp, targets_tmp
    
    if train:
        setname = 'train'
    else:
        setname = 'val'

    directory = os.path.join(root, setname)

    is_valid_file = None
    extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    classes, class_to_idx = find_classes(directory)
    samples, lb_index = make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
    if len(samples) == 0:
        msg = "Found 0 files in subfolders of: {}\n".format(directory)
        if extensions is not None:
            msg += "Supported extensions are: {}".format(",".join(extensions))
        raise RuntimeError(msg)
    
    data = [s[0] for s in samples]
    targets = [s[1] for s in samples]
    
    if train:
        labels_num = int(percentage*len(data)/len(classes))
        select_data, select_targets, u_data, u_targets = SelectfromClasses(data, targets, index, num_per_class=labels_num, return_ulb=return_ulb)
    else:
        select_data, select_targets = SelectfromClasses(data, targets, index)
    if return_ulb:
        return select_data, select_targets, u_data, u_targets
    return select_data, select_targets


def find_classes(directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(directory, class_to_idx, percentage=-1, extensions=None, is_valid_file=None, include_lb_to_ulb=True, lb_index=None):   
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return x.lower().endswith(extensions)
    
    lb_idx = {}
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            random.shuffle(fnames)
            if percentage != -1:
                fnames = fnames[:int(len(fnames) * percentage)]
            if percentage != -1:
                lb_idx[target_class] = fnames
            for fname in fnames:
                if not include_lb_to_ulb:
                    if fname in lb_index[target_class]:
                        continue
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    gc.collect()
    return instances, lb_idx


def get_label2id(filename):
    label_set = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line not in label_set.keys():
                label_set[line] = len(label_set)
    return label_set


def savepickle(data, file_path):
    mkdir_p(osp.dirname(file_path), delete=False)
    print('pickle into', file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def mkdir_p(path, delete=False, print_info=True):
    if path == '': return

    if delete:
        subprocess.call(('rm -r ' + path).split())
    if not osp.exists(path):
        if print_info:
            print('mkdir -p  ' + path)
        subprocess.call(('mkdir -p ' + path).split())


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def find_and_delete_max(tensor):
    original_shape = tensor.shape
    row_map = list(range(original_shape[0]))
    col_map = list(range(original_shape[1]))
    delete_sequence = []

    while tensor.numel() > 0:
        max_value = torch.max(tensor)
        max_idx = (tensor == max_value).nonzero(as_tuple=False)[0]
        row, col = max_idx[0].item(), max_idx[1].item()

        # 获取原始的行列坐标
        original_row, original_col = row_map[row], col_map[col]
        delete_sequence.append((original_row, original_col))

        # 删除行和列
        tensor = torch.cat((tensor[:row, :], tensor[row+1:, :]), dim=0)
        tensor = torch.cat((tensor[:, :col], tensor[:, col+1:]), dim=1)

        # 更新行列映射
        del row_map[row]
        del col_map[col]

    return delete_sequence


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    from torch.optim.lr_scheduler import LambdaLR
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an integer parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def generate_random_orthogonal_matrix(feat_in, num_classes):
    """生成随机正交矩阵"""
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)  # 使用QR分解生成正交矩阵
    orth_vec = torch.tensor(orth_vec).float()  # 转换为PyTorch张量
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "生成的矩阵不是正交矩阵"
    return orth_vec


def generate_etf_vector(in_channels, num_classes):
    """生成等距紧框架 (ETF) 向量"""
    # 生成正交矩阵
    orth_vec = generate_random_orthogonal_matrix(in_channels, num_classes)
    
    # 创建单位矩阵和全1矩阵
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    
    # 生成ETF向量
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    
    return etf_vec

@torch.no_grad()
def mixup_one_target(x, y, alpha=1.0, is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda
    """
    x, u = x.chunk(2, dim=0)
    y, p = y.chunk(2, dim=0)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * u[index]
    mixed_y = lam * y + (1 - lam) * p[index]
    return mixed_x, mixed_y, lam

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())