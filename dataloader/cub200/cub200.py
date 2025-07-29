import os
import os.path as osp

import pdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .autoaugment import AutoAugImageNetPolicy, _make_multicrop_imgnt_transforms


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class CUB200(Dataset):

    def __init__(self, root='/data/zhoudw/FSCIL', train=True,
                 index_path=None, index=None, base_sess=None, 
                 autoaug=1, use_conloss=False, use_extend_dataset=False, 
                 extend_dataset=None, extend_dataset_path=None, num_per_class_for_extend_dataset=None,
                 is_snn=False, is_support=False, support_views=1, multicrop=6, mc_size=96):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.classes = index
        self._pre_operate(self.root)
        
        if is_snn:
            if is_support:
                if train:
                    base_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    self.transform = ContrastiveLearningViewGenerator(base_transform=base_transform, n_views=support_views)
                    if base_sess:
                        self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                    else:
                        self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

                mint = None
                self.target_indices = []
                for t in range(len(self.classes)):
                    indices = np.where(np.array(self.targets) == t)[0].tolist()
                    self.target_indices.append(indices)
                    mint = len(indices) if mint is None else min(mint, len(indices))
                    print(f'num-labeled target {t} {len(indices)}')
                print(f'min. labeled indices {mint}')
            else:
                if train:
                    base_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        # AutoAugImageNetPolicy(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    self.transform = ContrastiveLearningViewGenerator(base_transform=base_transform)
                    self.multicrop, self.mc_transform = _make_multicrop_imgnt_transforms(multicrop, size=mc_size, normalize=True)
                    if base_sess:
                        self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                    else:
                        self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        else:
            if autoaug==0:
                #do not use autoaug
                if train:
                    base_transform = transforms.Compose([
                        transforms.Resize(256),
                        # transforms.CenterCrop(224),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        #add autoaug
                        #AutoAugImageNetPolicy(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    if use_conloss:
                        self.transform = ContrastiveLearningViewGenerator(base_transform=base_transform)
                    else:
                        self.transform = base_transform
                    # self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
                    if base_sess:
                        self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                    else:
                        self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                #use autoaug
                if train:
                    base_transform = transforms.Compose([
                        transforms.Resize(256),
                        # transforms.CenterCrop(224),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        #add autoaug
                        AutoAugImageNetPolicy(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    if use_conloss:
                        self.transform = ContrastiveLearningViewGenerator(base_transform=base_transform)
                    else:
                        self.transform = base_transform
                    # self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
                    if base_sess:
                        self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                    else:
                        self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            
        print('CUB200 dataset with {} samples'.format(len(self.data)))
        # extend dataset
        if use_extend_dataset:
            print('Using extend dataset: {}'.format(extend_dataset))
            extend_data, extend_targets = get_extend_dataset_data(extend_dataset, extend_dataset_path, num_per_class_for_extend_dataset)
            print('Extend dataset with {} samples'.format(len(extend_data)))
            self.data += extend_data
            extend_targets = extend_targets + np.max(self.targets) + 1
            self.targets += extend_targets.tolist()

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'CUB_200_2011/images.txt')
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train:
            for k in train_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectfromTxt(self, data2label, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.root, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = Image.open(path).convert('RGB')
        img = self.transform(Image.open(path).convert('RGB'))
        
        if hasattr(self, 'mc_transform') and self.mc_transform is not None and self.multicrop > 0:
            mc_imgs = [self.mc_transform(image) for _ in range(int(self.multicrop))]
            if isinstance(img, list):
                return *img, *mc_imgs, targets
            return img, *mc_imgs, targets
        if isinstance(img, list):
            return *img, targets
        return img, targets


class CUB200_concate(Dataset):
    def __init__(self, train,x1,y1,x2,y2):
        
        if train:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.data=x1+x2
        self.targets=y1+y2
        print(len(self.data),len(self.targets))

    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets


def get_extend_dataset_data(extend_dataset, extend_dataset_path, num_per_class=None):
    data = []
    targets = []
    if extend_dataset == 'miniimagenet':
        # get data and targets
        image_path = os.path.join(extend_dataset_path, 'images')
        csv_path = os.path.join(extend_dataset_path, 'train.csv')
        # read csv file
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip().split(',')
                data.append(os.path.join(image_path, line[1]))
                targets.append(int(line[2]))
        # select num_per_class samples per class
        if num_per_class is not None:
            targets = np.array(targets)
            new_data, new_targets = [], []
            for i in range(np.max(targets) + 1):
                ind_cl = np.where(i == targets)[0]
                np.random.shuffle(ind_cl)
                new_data.extend([data[j] for j in ind_cl[:num_per_class]])
                new_targets.extend([targets[j] for j in ind_cl[:num_per_class]])
            data, targets = new_data, new_targets
    else:
        raise ValueError('No such dataset: {}'.format(extend_dataset))
    return data, targets
    

# if __name__ == '__main__':
#     txt_path = "../../data/index_list/cub200/session_1.txt"
#     # class_index = open(txt_path).read().splitlines()
#     base_class = 100
#     class_index = np.arange(base_class)
#     dataroot = '/root/autodl-tmp/taicai/FSCIL/data/FSCIL'
#     batch_size_base = 32
#     # trainset = CUB200(root=dataroot, train=True,  index=class_index,
#     #                   base_sess=True, autoaug=0, is_snn=True, is_support=True, support_views=2)
#     trainset = CUB200(root=dataroot, train=True,  index=class_index,
#                       base_sess=True, autoaug=0, is_snn=True, is_support=False, mc_size=96, multicrop=6)
#     trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
#                                               pin_memory=True)
#     for batch_idx, udata in enumerate(trainloader):
#         print(batch_idx, len(udata), udata[-1], len(udata[0]), udata[1].shape)
#         break
    
    