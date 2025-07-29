from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance
from utils.randaugment import RandAugment, RandomResizedCropAndInterpolation
from utils.keep_autoaugment import CIFAR10Policy, SVHNPolicy, SubPolicy, ImageNetPolicy, Cutout
import torch
import math
import numpy as np
import copy
transformtypedict = dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        return out

class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)

        if transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, phase='train'):
        if phase == "train":
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        elif phase == "test":
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']
        elif phase == "reserved":
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'Normalize']
        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


def get_transform(phase, image_size, normalize_param):
    trans_loader = TransformLoader(image_size, normalize_param=normalize_param)
    if phase == "train":
        transform = trans_loader.get_composed_transform(phase)
    elif phase == "test":
        transform = trans_loader.get_composed_transform(phase)
    elif phase == "reserved":
        transform = trans_loader.get_composed_transform(phase)
    else:
        print("unknow phase")
        exit()
    return transform


class BaseDataset(Dataset):

    def __init__(self, phase, image_size, label2id, strong_transform=None, dataset='cub', autoaug=False):
        self.data = []
        self.targets = []
        self.dataset = dataset
        self.label2id = label2id
        if dataset == 'cifar100' or dataset == 'cifar10':
            if dataset == 'cifar100':
                normalize_param=dict(mean=[x / 255 for x in [129.3, 124.1, 112.4]], std=[x / 255 for x in [68.2, 65.4, 70.4]])
            else:
                normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if phase == 'train':
                if autoaug:
                    self.transform = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
                        transforms.RandomHorizontalFlip(),
                        CIFAR10Policy(),
                        transforms.ToTensor(),
                        Cutout(n_holes=1, length=16),
                        transforms.Normalize(**normalize_param)
                    ])
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
                        transforms.RandomHorizontalFlip(),
                        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                        # transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(**normalize_param)
                    ])
                
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            if strong_transform is None:
                self.strong_transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                    # transforms.RandomGrayscale(p=0.2),
                    RandAugment(3, 5),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            else:
                self.strong_transform = strong_transform
        elif dataset == 'miniimagenet':
            normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if phase == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((int(math.floor(image_size / 0.875)), int(math.floor(image_size / 0.875)))),
                    transforms.RandomCrop((image_size, image_size)),
                    ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(math.floor(math.floor(image_size / 0.875))),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            if strong_transform is None:
                self.strong_transform = transforms.Compose([
                    transforms.Resize((int(math.floor(image_size / 0.875)), int(math.floor(image_size / 0.875)))),
                    RandomResizedCropAndInterpolation((image_size, image_size)),
                    ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                    transforms.RandomHorizontalFlip(),
                    RandAugment(3, 10),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            else:
                self.strong_transform = strong_transform
        else:
            normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = get_transform(phase, image_size, normalize_param)
            # self.transform = get_transform(phase, image_size)
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(self.transform)
                # self.strong_transform.transforms.insert(0, RandAugment(3,5))
                self.strong_transform.transforms.insert(-2, RandAugment(3, 10))
            else:
                self.strong_transform = strong_transform

        # if dataset == 'cifar100':
        #     normalize_param=dict(mean=[x / 255 for x in [129.3, 124.1, 112.4]], std=[x / 255 for x in [68.2, 65.4, 70.4]])
        #     if phase == 'train':
        #         self.transform = transforms.Compose([
        #             transforms.Resize(image_size),
        #             transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.ToTensor(),
        #             transforms.Normalize(**normalize_param)
        #         ])
        #     else:
        #         self.transform = transforms.Compose([
        #             transforms.Resize(image_size),
        #             transforms.ToTensor(),
        #             transforms.Normalize(**normalize_param)
        #         ])
        # else:
        #     normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     self.transform = get_transform(phase, image_size, normalize_param, dataset=dataset)
        # # self.transform = get_transform(phase, image_size)
        # if strong_transform is None:
        #     self.strong_transform = copy.deepcopy(self.transform)
        #     self.strong_transform.transforms.insert(0, RandAugment(3,5))
        #     # self.strong_transform.transforms.insert(-2, RandAugment(3,5))
        # else:
        #     self.strong_transform = strong_transform

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        if self.dataset == 'cifar100' or self.dataset == 'cifar10':
            image = self.transform(Image.fromarray(path))
            image_s = self.strong_transform(Image.fromarray(path))
        else:
            image = self.transform(Image.open(path).convert('RGB'))
            image_s = self.strong_transform(Image.open(path).convert('RGB'))
        label = int(label)

        return image, image_s, label

    def __len__(self):
        return len(self.data)


class BaseDataset_flip(Dataset):

    def __init__(self, phase, image_size, label2id, dataset='cub', autoaug=False):
        self.data = []
        self.targets = []
        self.label2id = label2id
        self.dataset = dataset
        if dataset == 'cifar100' or dataset == 'cifar10':
            if dataset == 'cifar100':
                normalize_param=dict(mean=[x / 255 for x in [129.3, 124.1, 112.4]], std=[x / 255 for x in [68.2, 65.4, 70.4]])
            else:
                normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if phase == 'train':
                if autoaug:
                    self.transform = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
                        transforms.RandomHorizontalFlip(),
                        CIFAR10Policy(),
                        transforms.ToTensor(),
                        Cutout(n_holes=1, length=16),
                        transforms.Normalize(**normalize_param)
                    ])
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
                        transforms.RandomHorizontalFlip(),
                        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                        # transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(**normalize_param)
                    ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
        else:
            normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = get_transform(phase, image_size, normalize_param)
        # self.transform = get_transform(phase, image_size)

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        if self.dataset == 'cifar100' or self.dataset == 'cifar10':
            image = self.transform(Image.fromarray(path))
        else:
            image = self.transform(Image.open(path).convert('RGB'))
        image = torch.flip(image, [2])
        label = int(label)

        return image, label

    def __len__(self):
        return len(self.data)


class BaseDataset_flag(Dataset):

    def __init__(self, phase, image_size, label2id, strong_transform=None, dataset='cub', autoaug=False):
        self.data = []
        self.targets = []
        self.flags = []
        self.on_flags = []
        self.dataset = dataset
        self.label2id = label2id
        if dataset == 'cifar100' or dataset == 'cifar10':
            if dataset == 'cifar100':
                normalize_param=dict(mean=[x / 255 for x in [129.3, 124.1, 112.4]], std=[x / 255 for x in [68.2, 65.4, 70.4]])
            else:
                normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if phase == 'train':
                if autoaug:
                    self.transform = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
                        transforms.RandomHorizontalFlip(),
                        CIFAR10Policy(),
                        transforms.ToTensor(),
                        Cutout(n_holes=1, length=16),
                        transforms.Normalize(**normalize_param)
                    ])
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
                        transforms.RandomHorizontalFlip(),
                        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                        # transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(**normalize_param)
                    ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            if strong_transform is None:
                self.strong_transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                    # transforms.RandomGrayscale(p=0.2),
                    RandAugment(3, 5),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            else:
                self.strong_transform = strong_transform
        elif dataset == 'miniimagenet':
            normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if phase == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((int(math.floor(image_size / 0.875)), int(math.floor(image_size / 0.875)))),
                    transforms.RandomCrop((image_size, image_size)),
                    ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(math.floor(math.floor(image_size / 0.875))),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            if strong_transform is None:
                self.strong_transform = transforms.Compose([
                    transforms.Resize((int(math.floor(image_size / 0.875)), int(math.floor(image_size / 0.875)))),
                    RandomResizedCropAndInterpolation((image_size, image_size)),
                    ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                    transforms.RandomHorizontalFlip(),
                    RandAugment(3, 10),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            else:
                self.strong_transform = strong_transform
        else:
            normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = get_transform(phase, image_size, normalize_param)
            # self.transform = get_transform(phase, image_size)
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(self.transform)
                # self.strong_transform.transforms.insert(0, RandAugment(3,5))
                self.strong_transform.transforms.insert(-2, RandAugment(3,10))
            else:
                self.strong_transform = strong_transform

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        flags = self.flags[index]
        on_flags = self.on_flags[index]
        if self.dataset == 'cifar100' or self.dataset == 'cifar10':
            image = self.transform(Image.fromarray(path))
            image_s = self.strong_transform(Image.fromarray(path))
        else:
            image = self.transform(Image.open(path).convert('RGB'))
            image_s = self.strong_transform(Image.open(path).convert('RGB'))
        label = int(label)

        return index, image, image_s, label, flags, on_flags

    def __len__(self):
        return len(self.data)


class UnlabelDataset(Dataset):

    def __init__(self, image_size, unlabeled_num=None, strong_transform=None, dataset='cub', autoaug=False):
        self.data = []
        self.targets = []
        self.dataset = dataset
        if dataset == 'cifar100' or dataset == 'cifar10':
            if dataset == 'cifar100':
                normalize_param=dict(mean=[x / 255 for x in [129.3, 124.1, 112.4]], std=[x / 255 for x in [68.2, 65.4, 70.4]])
            else:
                normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if autoaug:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    Cutout(n_holes=1, length=16),
                    transforms.Normalize(**normalize_param)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                    # transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            if strong_transform is None:
                self.strong_transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                    # transforms.RandomGrayscale(p=0.2),
                    RandAugment(3, 5),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            else:
                self.strong_transform = strong_transform
        elif dataset == 'miniimagenet':
            normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([
                transforms.Resize((int(math.floor(image_size / 0.875)), int(math.floor(image_size / 0.875)))),
                transforms.RandomCrop((image_size, image_size)),
                ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize_param)
            ])
            if strong_transform is None:
                self.strong_transform = transforms.Compose([
                    transforms.Resize((int(math.floor(image_size / 0.875)), int(math.floor(image_size / 0.875)))),
                    RandomResizedCropAndInterpolation((image_size, image_size)),
                    ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                    transforms.RandomHorizontalFlip(),
                    RandAugment(3, 10),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_param)
                ])
            else:
                self.strong_transform = strong_transform
        else:
            normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = get_transform('train', image_size, normalize_param)
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(self.transform)
                # self.strong_transform.transforms.insert(0, RandAugment(3,5))
                self.strong_transform.transforms.insert(-2, RandAugment(3,10))
            else:
                self.strong_transform = strong_transform

        # if dataset == 'cifar100':
        #     normalize_param=dict(mean=[x / 255 for x in [129.3, 124.1, 112.4]], std=[x / 255 for x in [68.2, 65.4, 70.4]])
        #     self.transform = transforms.Compose([
        #         transforms.Resize(image_size),
        #         transforms.RandomCrop(image_size, padding=int(image_size * (1 - 0.875)), padding_mode='reflect'),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(**normalize_param)
        #     ])
        # else:
        #     normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     self.transform = get_transform('train', image_size, normalize_param, dataset=dataset)
        # # self.transform = get_transform("train", image_size)
        # if strong_transform is None:
        #     self.strong_transform = copy.deepcopy(self.transform)
        #     self.strong_transform.transforms.insert(0, RandAugment(3,5))
        #     # self.strong_transform.transforms.insert(-2, RandAugment(3,5))
        # else:
        #     self.strong_transform = strong_transform
        
        if unlabeled_num != -1 and unlabeled_num is not None:
            try:
                self.data = self.data[: unlabeled_num]
                self.targets = self.targets[: unlabeled_num]
            except:
                pass
        

    def __getitem__(self, index):
        path, label = self.data[index], self.targets[index]
        if self.dataset == 'cifar100' or self.dataset == 'cifar10':
            image = self.transform(Image.fromarray(path))
            image_s = self.strong_transform(Image.fromarray(path))
        else:
            image = self.transform(Image.open(path).convert('RGB'))
            image_s = self.strong_transform(Image.open(path).convert('RGB'))
        return image, image_s, label

    def __len__(self):
        return len(self.data)


class ReservedUnlabelDataset(Dataset):

    def __init__(self, image_size, unlabeled_num=None):
        self.data = []
        self.label = []
        # self.transform = get_transform("reserved", image_size)

        if unlabeled_num != -1 and unlabeled_num is not None:
            try:
                self.data = self.data[: unlabeled_num]
            except:
                pass

    def __getitem__(self, index):
        # image = self.data[index]
        # image = self.transform(Image.open(path).convert('RGB'))
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)



