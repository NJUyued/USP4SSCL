import numpy as np
import torch
from dataloader.sampler import CategoriesSampler, ClassStratifiedSampler

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    

    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    
    if args.dataset == 'imagenet100':
        import dataloader.imagenet100.ImageNet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9


    if args.dataset == 'imagenet1000':
        import dataloader.imagenet1000.ImageNet as Dataset
        args.base_class = 600
        args.num_classes=1000
        args.way = 50
        args.shot = 5
        args.sessions = 9

    args.Dataset=Dataset
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True, index=class_index, 
                                         base_sess=True, use_conloss=args.use_conloss)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        if args.use_snn:
            trainset_sup = args.Dataset.CUB200(root=args.dataroot, train=True, index=class_index, base_sess=True, 
                                               autoaug=0, is_snn=True, is_support=True, support_views=args.supervised_views)
            trainset_unsup = args.Dataset.CUB200(root=args.dataroot, train=True, index=class_index, base_sess=True, 
                                                 autoaug=0, is_snn=True, is_support=False, mc_size=96, multicrop=args.multicrop)
        else:
            trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index=class_index, 
                                        base_sess=True, use_conloss=args.use_conloss,
                                        use_extend_dataset=args.use_extend_dataset, 
                                        extend_dataset=args.extend_dataset, 
                                        extend_dataset_path=args.extend_dataset_path, 
                                        num_per_class_for_extend_dataset=args.num_per_class_for_extend_dataset)
        
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index=class_index, base_sess=True, use_conloss=args.use_conloss)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True, use_conloss=args.use_conloss)
        testset = args.Dataset.ImageNet(root=args.dataroot, train=False, index=class_index)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    if args.use_snn:
        sampler = ClassStratifiedSampler(data_source=trainset_sup,
                                         world_size=1,
                                         rank=0,
                                         batch_size=args.sup_imgs_per_class,
                                         classes_per_batch=args.classes_per_batch,
                                         unique_classes=True,
                                         seed=args.seed, 
                                         )
        trainloader_sup = torch.utils.data.DataLoader(dataset=trainset_sup, batch_sampler=sampler, num_workers=8,
                                                      pin_memory=True)
        trainloader = torch.utils.data.DataLoader(dataset=trainset_unsup, batch_size=args.batch_size_base, shuffle=True,
                                                  drop_last=True,
                                                num_workers=8, pin_memory=True)
        return trainset_sup, trainloader, trainloader_sup, testloader
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                                num_workers=8, pin_memory=True)
    
        return trainset, trainloader, testloader

def get_base_dataloader_meta(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index)


    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False, use_conloss=args.use_conloss)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                        index_path=txt_path, use_conloss=args.use_conloss)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path, use_conloss=args.use_conloss)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path, use_conloss=args.use_conloss)

    if args.batch_size_new == 0:
            batch_size_new = trainset.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)
    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        testset = args.Dataset.ImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, None, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list