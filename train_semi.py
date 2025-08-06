import os
import sys
import copy
import torch
import argparse
import numpy as np
import utils_pytorch
import torch.nn as nn
import torch.optim as optim
from resnet import resnet18
from resnet20_cifar import resnet20
from resnet32_cifar import resnet32
from torch.optim import lr_scheduler
from dataloder import BaseDataset, BaseDataset_flag
from utils_incremental.compute_accuracy import compute_accuracy
from utils_incremental.compute_features import compute_features, compute_feats
from utils_incremental.incremental_train_and_eval_semi import incremental_train_and_eval, fill_pro_list, get_proto

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cub', type=str)
parser.add_argument('--num_classes', default=200, type=int)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--data_dir', default='dataset', type=str)
parser.add_argument('--nb_cl_fg', default=100, type=int, help='the number of classes in first session')
parser.add_argument('--nb_cl', default=10, type=int, help='Classes per group')
parser.add_argument('--nb_protos', default=20, type=int, help='Number of prototypes per class at the end')
parser.add_argument('--k_shot', default=5, type=int, help='')
parser.add_argument('--ckp_prefix', default=os.path.basename(sys.argv[0])[:-3], type=str, help='Checkpoint prefix')
parser.add_argument('--epochs', default=160, type=int, help='Epochs for first sesssion')
parser.add_argument('--T', default=2, type=float, help='Temperature for distialltion')
parser.add_argument('--beta', default=0.25, type=float, help='Beta for distialltion')
parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument('--rs_ratio', default=0.0, type=float, help='The ratio for resample')
parser.add_argument('--model_path', default='the path to resumed model', type=str)
parser.add_argument('--cm_path', default='the path to class means', type=str)
parser.add_argument('--unlabeled_iteration', default=100, type=int, help='the total iteration to add unlabeled data')
parser.add_argument('--update_unlabeled', action='store_true', default=True, help='if using selected unlabled data to update the class_mean')
parser.add_argument('--use_nearest_mean', action='store_true', default=True, help='if using nearest-mean-of-examplars classification for selecting unlabeled data')
parser.add_argument('--unlabeled_num', default=300, type=int, help='The total number for resample')
parser.add_argument('--unlabeled_num_selected', default=160, type=int, help='The number of selected unlabeled data')
parser.add_argument('--random_seed', default=1993, type=int, help='random seed')
parser.add_argument('--method', default='self_train', type=str, choices=['self_train', 'random', 'consistency'], help='the method for adding unlabeled data')
parser.add_argument('--uncertainty_distillation', action='store_true', default=False, help='if uncertainty distillation')
parser.add_argument('--flip_on_means', action='store_true', default=False, help='if flip when computing class-means')
parser.add_argument('--base_lamda', default=2, type=int, help='the base weight for distillation loss')
parser.add_argument('--u_t', default=3/5, type=int, help='the threshold in uncertainty estimation')
parser.add_argument('--adapt_lamda', action='store_true', default = False, help='adaptive weight for distillation loss')
parser.add_argument('--frozen_backbone_part', action='store_true', default = False, help='if freeze part of the backbone')
parser.add_argument('--include_neglabels', action='store_true', default = False, help='weather use neglabels')
parser.add_argument('--gpu', default=0, type=int, help='chose the gpu')
# add args
parser.add_argument('--use_conloss', action='store_true', default = False, help='weather use neglabels')
parser.add_argument('--epochs_new', default=60, type=int, help='Epochs for first sesssion')
parser.add_argument('--use_proto', action='store_true', default = False, help='weather use neglabels')
parser.add_argument('--update_proto', action='store_true', default = False, help='weather use neglabels')
parser.add_argument('--u_ratio', default=1, type=int, help='Epochs for first sesssion')
parser.add_argument('--u_iter', default=100, type=int, help='Epochs for first sesssion')
parser.add_argument('--lambda_kd', default=1.0, type=float, help='weather use neglabels')
parser.add_argument('--lambda_con', default=1.0, type=float, help='weather use neglabels')
parser.add_argument('--lambda_cons', default=1.0, type=float, help='weather use neglabels')
parser.add_argument('--lambda_reg', default=1.0, type=float, help='weather use neglabels')
parser.add_argument('--lambda_in', default=1.0, type=float, help='weather use neglabels')
parser.add_argument('--lambda_cat', default=1.0, type=float, help='weather use neglabels')
parser.add_argument('--lambda_ukd', default=1.0, type=float, help='weather use neglabels')
parser.add_argument("--base_lr", default=1e-3, type=float, help="Initial learning rate")
parser.add_argument("--new_lr", default=5e-4, type=float, help="Initial learning rate")
parser.add_argument('--train_batch_size', default=32, type=int, help='Epochs for first sesssion')
parser.add_argument('--test_batch_size', default=32, type=int, help='Epochs for first sesssion')
parser.add_argument('--kd_only_old', action='store_true', default = False, help='weather use neglabels')
parser.add_argument('--no_use_conloss_on_ulb', action='store_true', default = False, help='weather use neglabels')
parser.add_argument('--dim', default=512, type=int,)
parser.add_argument('--unlabels_predict_mode', default='sqeuclidean', type=str, choices=['sqeuclidean', 'cosine'],)
parser.add_argument("--use_ulb_kd", action='store_true', default=False,)
parser.add_argument("--use_lb_kd", action='store_true', default=False,)
parser.add_argument("--use_pretrain", action='store_true', default=False,)
parser.add_argument('--schedule', default='Milestone', type=str, choices=['step', 'Milestone', 'cosine'], help='the method for adding unlabeled data')
parser.add_argument('--model', default='resnet18', type=str, choices=['resnet32', 'resnet20', 'resnet18'],)
parser.add_argument('--proto_dim', default=512, type=int,)
# 未使用
parser.add_argument('--prompt_idx_pos', default=1, type=int, help='the positive prompt templat')
parser.add_argument('--prompt_idx_neg', default=1, type=int, help='the negtive prompt template')
parser.add_argument('--use_exclude', action='store_true', default = False, help='weather use neglabels')
parser.add_argument('--neg_topk', default=100, type=int, help='the negtive prompt template')
parser.add_argument('--con_margin', default=0.2, type=float, help='The ratio for resample')
parser.add_argument('--hard_negative', action='store_true', default = False, help='weather use neglabels')
parser.add_argument('--include_unlabel', action='store_true', default = False,help='weather use unlabels data to align text feature feace')
parser.add_argument('--use_da', action='store_true', default = False, help='weather use neglabels')
parser.add_argument('--use_class_weight', action='store_true', default = False, help='weather use neglabels')
parser.add_argument('--no_linear', action='store_true', default = False,)
parser.add_argument("--no_trans", action='store_true', default = False,)
parser.add_argument("--use_proto_classifer", action='store_true', default = False,)
parser.add_argument("--temperature", default=10.0, type=float, help="temperature")
parser.add_argument("--use_session_means", action='store_true', default = False,)
parser.add_argument('--warmup_epochs', default=60, type=int,)
parser.add_argument("--p_cutoff", default=0.95, type=float,)
parser.add_argument("--q_cutoff", default=0.25, type=float,)
parser.add_argument('--use_sim', action='store_true', default = False,)
parser.add_argument("--autoaug", action='store_true', default=False,)
parser.add_argument('--use_srd', action='store_true', default=False,)
parser.add_argument('--use_session_labels', action='store_true', default=False,)
parser.add_argument('--buffer_size', default=500, type=int, help='Epochs for first sesssion')
parser.add_argument('--use_feats_kd', action='store_true', default=False,)
parser.add_argument('--use_ulb_aug', action='store_true', default=False,)
parser.add_argument('--adapt_weight', action='store_true', default=False,)
parser.add_argument('--use_mix_up', action='store_true', default=False,)
parser.add_argument("--mixup_alpha", default=0.95, type=float,)
parser.add_argument('--use_hard_labels', action='store_true', default=False,)
parser.add_argument('--use_old', action='store_true', default=False,)
parser.add_argument('--use_metric_loss', action='store_true', default=False,)
parser.add_argument('--kd_mode', default='logits', type=str, choices=['logits', 'feats', 'attention', 'logits_at'],)
parser.add_argument('--ulb_kd_mode', default='logits', type=str, choices=['logits', 'feats', 'attention', 'cosine', 'similarity'],)
parser.add_argument('--use_adv', action='store_true', default=False,)
parser.add_argument('--proto_clissifier', action='store_true', default=False,)
parser.add_argument('--percentage', default=0.01, type=float, help='')
parser.add_argument('--adapt_filled', action='store_true', default=False,)
parser.add_argument('--is_fewshot', action='store_true', default=False,)
parser.add_argument('--no_use_filled', action='store_true', default=False,)
parser.add_argument('--onlytest', action='store_true', default=False,)
parser.add_argument('--ckp_name', default='the path to resumed model', type=str)
parser.add_argument('--best_ckp_name', default='the path to resumed model', type=str)

args = parser.parse_args()
assert (args.nb_cl_fg % args.nb_cl == 0)
assert (args.nb_cl_fg >= args.nb_cl)
test_batch_size = args.test_batch_size  # Batch size for test (original 100)
eval_batch_size = 32  # Batch size for eval
train_batch_size = args.train_batch_size  # Batch size for train
base_lr = args.base_lr # 1e-3 # Initial learning rate
lr_strat = [60, 120, 170]  # Epochs where learning rate gets decreased
lr_factor = 0.1 # Learning rate decrease factor
custom_weight_decay = 5e-4  # Weight Decay
lr_strat_new = [80, 120, 150]
custom_weight_decay_new = 2e-4
custom_momentum = 0.9  # Momentum
args.ckp_prefix = '{}_nb_cl_fg_{}_nb_cl_{}_nb_protos_{}'.format(args.ckp_prefix, args.nb_cl_fg, args.nb_cl, args.nb_protos)

np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

print(args)

if args.dataset == 'cub':
    dictionary_size = 30
    label2id = utils_pytorch.get_label2id("./dataset/cub/split/label_name.txt")
    id2label = {index: la for la, index in label2id.items()}
elif args.dataset == 'cifar10':
    dictionary_size = 5000
    label2id, id2label = None, None
elif args.dataset == 'cifar100':
    dictionary_size = 500
    label2id, id2label = None, None
elif args.dataset == 'miniimagenet': 
    dictionary_size = 500 
    label2id, id2label = None, None   
elif args.dataset == 'imagenet100':
    dictionary_size = 1300
    label2id, id2label = None, None  
    
order_name = "./checkpoint/seed_{}_{}_order_run.pkl".format(args.random_seed, args.dataset)
print("Order name: {}".format(order_name))
order = np.arange(args.num_classes)
order_list = list(order)
print(order_list)

X_valid_cumuls = []
X_protoset_cumuls = []
X_train_cumuls = []
Y_valid_cumuls = []
Y_protoset_cumuls = []
Y_train_cumuls = []

X_valid_cumuls_base = []
Y_valid_cumuls_base = []
X_valid_cumul_novel = []
Y_valid_cumul_novel = []

# alpha_dr_herding = np.zeros((int(args.num_classes / args.nb_cl), dictionary_size, args.nb_cl), np.float32)

# The following contains all the training samples of the different classes
# because we want to compare our method with the theoretical case where all the training samples are stored
prototypes = [[] for i in range(args.num_classes)]
prototypes_flag = [[] for i in range(args.num_classes)]
prototypes_on_flag = [[] for i in range(args.num_classes)]

start_session = int(args.nb_cl_fg / args.nb_cl) - 1

alpha_dr_herding = []

# 获得所有类别的 ETF 特征表示，用于模型训练时对齐text特征空间
text_anchor = None
etf_vec = utils_pytorch.generate_etf_vector(args.dim, args.num_classes)
# 保存生成的ETF特征表示
# np.save('etf_vec_{}_{}.npy'.format(args.dim, args.ckp_prefix), etf_vec.T)

text_anchor = torch.tensor(etf_vec.T).to(device)
print('ETF anchor shape: {}'.format(text_anchor.shape))

for session in range(start_session, int(args.num_classes / args.nb_cl)):
    new_classes_names = list()
    
    # 针对持续学习的模型参数变换
    if session == start_session:
        # args.rs_ratio = 0.2
        ############################################################
        last_iter = 0
        ############################################################
        if args.resume:
            print('resume the results of first session')
            ckp_name = args.model_path
            if args.model == 'resnet20':
                print("resnet20")
                tg_model = resnet20(num_classes=args.nb_cl_fg, pretrained=args.use_pretrain, 
                                    use_proto_classifer=args.use_proto_classifer,
                                    no_trans=args.no_trans, temperature=args.temperature,
                                    dim=args.dim, no_linear=args.no_linear)
            elif args.model == 'resnet18':
                print('resnet18')
                tg_model = resnet18(num_classes=args.nb_cl_fg, pretrained=args.use_pretrain, 
                                    use_proto_classifer=args.use_proto_classifer, 
                                    no_trans=args.no_trans, temperature=args.temperature,
                                    dim=args.dim, no_linear=args.no_linear)
            elif args.model == 'resnet32':
                print('resnet32')
                tg_model = resnet32(num_classes=args.nb_cl_fg, pretrained=args.use_pretrain, 
                                    use_proto_classifer=args.use_proto_classifer,
                                    no_trans=args.no_trans, temperature=args.temperature,
                                    dim=args.dim, no_linear=args.no_linear)

            else:
                raise ValueError('model {} not supported'.format(args.model))

            tg_model = torch.load(ckp_name)
            ref_model = None
            args.epochs = 0
        else:
            if args.use_pretrain:
                print('load the pretrained model')
            if args.model == 'resnet20':
                print("resnet20")
                tg_model = resnet20(num_classes=args.nb_cl_fg, pretrained=args.use_pretrain, 
                                    use_proto_classifer=args.use_proto_classifer,
                                    no_trans=args.no_trans, temperature=args.temperature,
                                    dim=args.dim, no_linear=args.no_linear)
            elif args.model == 'resnet18':
                print("resnet18")
                tg_model = resnet18(num_classes=args.nb_cl_fg, pretrained=args.use_pretrain,
                                    use_proto_classifer=args.use_proto_classifer, 
                                    no_trans=args.no_trans, temperature=args.temperature,
                                    dim=args.dim, no_linear=args.no_linear)
            elif args.model == 'resnet32':
                print("resnet32")
                tg_model = resnet32(num_classes=args.nb_cl_fg, pretrained=args.use_pretrain, 
                                    use_proto_classifer=args.use_proto_classifer,
                                    no_trans=args.no_trans, temperature=args.temperature,
                                    dim=args.dim, no_linear=args.no_linear)
            else:
                raise ValueError('model {} not supported'.format(args.model))
            
            ref_model = None
    else:
        # args.rs_ratio = 0.99
        last_iter = session
        ############################################################
        # increment classes
        ref_model = copy.deepcopy(tg_model)
        in_features = tg_model.fc.in_features
        out_features = tg_model.fc.out_features
        if args.use_proto_classifer:
            new_fc = nn.Linear(in_features, out_features + args.nb_cl, bias=False)
            new_fc.weight.data[:out_features] = tg_model.fc.weight.data
        else:    
            new_fc = nn.Linear(in_features, out_features + args.nb_cl)
            new_fc.weight.data[:out_features] = tg_model.fc.weight.data
            new_fc.bias.data[:out_features] = tg_model.fc.bias.data
        tg_model.fc = new_fc

        # session fc param set
        if args.use_session_labels:
            if tg_model.fc_session is None:
                tg_model.fc_session = nn.Linear(in_features, 2)
            else:
                out_session_features = tg_model.fc_session.out_features
                new_session_fc = nn.Linear(in_features, out_features+1)
                new_session_fc.weight.data[:out_session_features] = tg_model.fc_session.weight.data
                new_session_fc.bias.data[:out_session_features] = tg_model.fc_session.bias.data
                tg_model.fc_session = new_session_fc
    
    # 准备数据
    unlabeled_data = None
    unlabeled_gt = None

    if args.dataset == 'cub':
        train_file = os.path.join(args.data_dir, args.dataset, "split", "session_{}.txt".format(session - start_session+1))
        test_file = os.path.join(args.data_dir, args.dataset, "split", "test_{}.txt".format(session - start_session+1))
        X_train, Y_train = utils_pytorch.get_data_file(train_file, "./dataset/cub/", label2id)
        X_valid,  Y_valid = utils_pytorch.get_data_file(test_file, "./dataset/cub/", label2id)
    
    elif args.dataset == 'miniimagenet':
        train_file = os.path.join(args.data_dir, args.dataset, "split", "session_{}.txt".format(session - start_session+1))
        test_file = os.path.join(args.data_dir, args.dataset, "split", "test_{}.txt".format(session - start_session+1))
        X_train, Y_train = utils_pytorch.get_data_file_miniimagenet(root="./dataset/miniimagenet/", base_session=False, index=train_file, train=True, unlabel=False)
        X_valid,  Y_valid = utils_pytorch.get_data_file_miniimagenet(root="./dataset/miniimagenet/", base_session=False, index=test_file, train=False, unlabel=False)

    elif args.dataset == 'cifar10':
        # 随机数据集划分
        class_index = np.arange(session * args.nb_cl, (session + 1) * args.nb_cl)
        X_train, Y_train, unlabeled_data, unlabeled_gt = utils_pytorch.get_data_file_cifar(data_dir="./cifar10/", base_session=True, index=class_index, train=True, unlabel=False, labels_num=args.k_shot, return_ulb=True, dataset=args.dataset)
        X_valid,  Y_valid = utils_pytorch.get_data_file_cifar(data_dir="./cifar10/", base_session=True, index=class_index, train=False, unlabel=False, dataset=args.dataset)
        
    elif args.dataset == 'cifar100':
        # 随机数据集划分
        class_index = np.arange(session * args.nb_cl, (session + 1) * args.nb_cl)
        X_train, Y_train, unlabeled_data, unlabeled_gt = utils_pytorch.get_data_file_cifar(data_dir="./cifar100/", base_session=True, index=class_index, train=True, unlabel=False, labels_num=args.k_shot, return_ulb=True, dataset=args.dataset)
        X_valid, Y_valid = utils_pytorch.get_data_file_cifar(data_dir="./cifar100/", base_session=True, index=class_index, train=False, unlabel=False)
        
    elif args.dataset == 'imagenet100':
        class_index = np.arange(session * args.nb_cl, (session + 1) * args.nb_cl)
        X_train, Y_train, unlabeled_data, unlabeled_gt = utils_pytorch.get_data_file_imagenet100(root="./dataset/imagenet100/", base_session=True, index=class_index, train=True, unlabel=False, percentage=args.percentage, return_ulb=True, dataset=args.dataset)
        X_valid, Y_valid = utils_pytorch.get_data_file_imagenet100(root="./dataset/imagenet100/", base_session=True, index=class_index, train=False, unlabel=False)

    # 针对持续少样本学习的数据准备过程
    if session > start_session and args.is_fewshot:
        if args.dataset == 'cub':
            unlabeled_file = os.path.join(args.data_dir, args.dataset, "split", "unlabeled_{}.txt".format(session - start_session+1))
            unlabeled_data, unlabeled_gt = utils_pytorch.get_data_file(unlabeled_file, "./dataset/cub/", label2id, unlabel=False)  #unlabeled=True
        elif args.dataset == 'cifar100':
            train_file = os.path.join(args.data_dir, args.dataset, "split", "session_{}.txt".format(session - start_session+1))
            class_train_index = open(train_file).read().splitlines()
            class_index = np.arange(session * args.nb_cl, (session + 1) * args.nb_cl)
            unlabeled_data, unlabeled_gt = utils_pytorch.get_data_file_cifar(data_dir="./dataset/cifar100/", base_session=False, index=class_train_index, train=True, unlabel=True, class_list=class_index)
        elif args.dataset == 'miniimagenet':
            txt_path = os.path.join(args.data_dir, args.dataset, "split", "session_{}.txt".format(session - start_session+1))
            class_index = np.arange(session * args.nb_cl, (session + 1) * args.nb_cl)
            unlabeled_data, unlabeled_gt = utils_pytorch.get_data_file_miniimagenet(root="./dataset/miniimagenet/", base_session=False, index=txt_path, train=True, unlabel=True, class_list=class_index)    

    if isinstance(X_train, list):
        X_train = np.array(X_train)
    if isinstance(Y_train, list):
        Y_train = np.array(Y_train)
    if isinstance(X_valid, list):
        X_valid = np.array(X_valid)
    if isinstance(Y_valid, list):
        Y_valid = np.array(Y_valid)
    if isinstance(unlabeled_data, list):
        unlabeled_data = np.array(unlabeled_data)
    if isinstance(unlabeled_gt, list):
        unlabeled_gt = np.array(unlabeled_gt)

    if args.unlabeled_num == 0:
        unlabeled_data=None
        unlabeled_gt=None
    elif args.unlabeled_num == -1:
        unlabeled_data=unlabeled_data
        unlabeled_gt=unlabeled_gt
    else:
        try:
            unlabeled_data = unlabeled_data[:args.unlabeled_num]
            unlabeled_gt = unlabeled_gt[:args.unlabeled_num]
        except:
            unlabeled_data = unlabeled_data
            unlabeled_gt == unlabeled_gt
    
    print("session: {}, X_train size: {}, X_valid size: {}".format(session, X_train.shape, X_valid.shape))
    print('Max and Min of train labels: {}, {}'.format(min(Y_train), max(Y_train)))
    print('Max and Min of valid labels: {}, {}'.format(min(Y_valid), max(Y_valid)))
    if unlabeled_gt is not None and len(unlabeled_gt) > 0:
        print("session: {}, ULX_train shape: {}, ULX_train shape: {}".format(session, unlabeled_data.shape, unlabeled_gt.shape))
        print('Max and Min of unlabel train labels: {}, {}'.format(min(unlabeled_gt), max(unlabeled_gt)))

    if args.is_fewshot:
        if session == start_session:
            for orde in range(0, (session + 1) * args.nb_cl):
                prototypes[orde] = X_train[np.where(Y_train == order[orde])]
                prototypes_flag[orde] = np.ones(len(prototypes[orde]), dtype = int)
                if orde < args.nb_cl_fg:
                    prototypes_on_flag[orde] = np.ones(len(prototypes[orde]), dtype=int)
                else:
                    prototypes_on_flag[orde] = np.zeros(len(prototypes[orde]), dtype=int)
        else:
            for orde in range(session * args.nb_cl, (session + 1) * args.nb_cl):
                prototypes[orde] = X_train[np.where(Y_train == order[orde])]
                prototypes_flag[orde] = np.ones(len(prototypes[orde]), dtype = int)
                if orde < args.nb_cl_fg:
                    prototypes_on_flag[orde] = np.ones(len(prototypes[orde]), dtype=int)
                else:
                    prototypes_on_flag[orde] = np.zeros(len(prototypes[orde]), dtype=int)
    else:
        for orde in range(session * args.nb_cl, (session + 1) * args.nb_cl):
            prototypes[orde] = X_train[np.where(Y_train == order[orde])]
            prototypes_flag[orde] = np.ones(len(prototypes[orde]), dtype = int)
            if orde < args.nb_cl_fg:
                prototypes_on_flag[orde] = np.ones(len(prototypes[orde]), dtype=int)
            else:
                prototypes_on_flag[orde] = np.zeros(len(prototypes[orde]), dtype=int)

    X_train_cumuls.append(X_train)
    X_train_cumul = np.concatenate(X_train_cumuls)
    Y_train_cumuls.append(Y_train)
    Y_train_cumul = np.concatenate(Y_train_cumuls)
    
    X_valid_cumuls.append(X_valid)
    X_valid_cumul = np.concatenate(X_valid_cumuls)
    Y_valid_cumuls.append(Y_valid)
    Y_valid_cumul = np.concatenate(Y_valid_cumuls)
    
    if session == start_session:
        X_flag = []
        X_on_flag = []
        for cls_id in range(0, (session + 1) * args.nb_cl):
            X_flag = np.append(X_flag, prototypes_flag[cls_id])
            X_on_flag = np.append(X_on_flag, prototypes_on_flag[cls_id])

        X_valid_cumuls_base.append(X_valid)
        Y_valid_cumuls_base.append(Y_valid)
        X_valid_cumul_base = np.concatenate(X_valid_cumuls_base)
        Y_valid_cumul_base = np.concatenate(Y_valid_cumuls_base)
    else:
        X_protoset = np.concatenate(X_protoset_cumuls)
        Y_protoset = np.concatenate(Y_protoset_cumuls)
        X_protoset_flag = np.concatenate(X_protoset_cumuls_flag)
        X_protoset_on_flag = np.concatenate(X_protoset_cumuls_on_flag)
        X_current_flag = []
        X_current_on_flag = []
        for cls_id in range(session * args.nb_cl, (session + 1) * args.nb_cl):
            X_current_flag = np.append(X_current_flag, prototypes_flag[cls_id])
            X_current_on_flag = np.append(X_current_on_flag, prototypes_on_flag[cls_id])
        X_current_flag = np.array(X_current_flag)
        X_current_on_flag = np.array(X_current_on_flag)

        if args.rs_ratio > 0:
            # 1/rs_ratio = (len(X_train)+len(X_protoset)*scale_factor)/(len(X_protoset)*scale_factor)
            scale_factor = (len(X_train) * args.rs_ratio) / (len(X_protoset) * (1 - args.rs_ratio))
            rs_sample_weights = np.concatenate((np.ones(len(X_train)), np.ones(len(X_protoset)) * scale_factor))
            # number of samples per epoch, undersample on the new classes
            # rs_num_samples = len(X_train) + len(X_protoset)
            rs_num_samples = int(len(X_train) / (1 - args.rs_ratio))
            print("X_train:{}, X_protoset:{}, rs_num_samples:{}".format(len(X_train), len(X_protoset), rs_num_samples))

        X_train = np.concatenate((X_train, X_protoset), axis=0)
        Y_train = np.concatenate((Y_train, Y_protoset))
        X_flag = np.concatenate((X_protoset_flag, X_current_flag))
        X_on_flag = np.concatenate((X_protoset_on_flag, X_current_on_flag))

        if len(X_valid_cumul_novel) != 0:
            X_valid_cumuls_base.append(X_valid_cumul_novel)
            Y_valid_cumuls_base.append(Y_valid_cumul_novel)
            X_valid_cumul_base = np.concatenate(X_valid_cumuls_base)
            Y_valid_cumul_base = np.concatenate(Y_valid_cumuls_base)

        X_valid_cumul_novel = X_valid
        Y_valid_cumul_novel = Y_valid
        ###################

    # 针对持续学习的数据准备过程
    if session > start_session:
        base_lr = args.new_lr # 0.0005
        args.epochs = args.epochs_new
        custom_weight_decay = custom_weight_decay_new
        print('the learning rate is {}'.format(base_lr))

    print('Batch of classes number {0} arrives ...'.format(session))
    ############################################################
    # 准备数据集
    trainset = BaseDataset_flag("train", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
    trainset.data = X_train
    trainset.targets = Y_train
    trainset.flags = X_flag
    trainset.on_flags = X_on_flag

    if session > start_session and args.rs_ratio > 0 and scale_factor > 1:
        index1 = np.where(rs_sample_weights > 1)[0]
        index2 = np.where(Y_train < session * args.nb_cl)[0]
        assert ((index1 == index2).all())
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(rs_sample_weights, 
                                                                       rs_num_samples)
        if len(trainset) < train_batch_size:
            train_batch_size = len(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                  shuffle=False, sampler=train_sampler, num_workers=4)
    else:
        if len(trainset) < train_batch_size:
            train_batch_size = len(trainset)
        sampler_x = torch.utils.data.RandomSampler(trainset, replacement=True, num_samples = args.u_iter * train_batch_size)
        batch_sampler_x = torch.utils.data.BatchSampler(sampler_x, train_batch_size, drop_last=True)  # yield a batch of samples one time        
        trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler_x, num_workers=4)
                        
    testset = BaseDataset("test", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
    testset.data = X_valid_cumul
    testset.targets = Y_valid_cumul
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    if args.include_unlabel and (not args.is_fewshot or session > start_session):
        unlabeled_trainset = BaseDataset_flag("test", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
        unlabeled_trainset.data = unlabeled_data
        unlabeled_trainset.targets = unlabeled_gt
        unlabeled_trainset.flags = np.ones(len(unlabeled_data), dtype=int)
        unlabeled_trainset.on_flags = np.ones(len(unlabeled_data), dtype=int)
        ssl_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, 
                                                    batch_size=args.u_ratio*train_batch_size, 
                                                    shuffle=True, num_workers=4, drop_last=False) 
    else:
        ssl_trainloader = None
    
    print("session: {}, dataset size: {}".format(session, len(trainloader.dataset)))
    print('Max and Min of train labels: {}, {}'.format(min(Y_train), max(Y_train)))
    print('Max and Min of valid labels: {}, {}'.format(min(Y_valid_cumul), max(Y_valid_cumul)))
    
    ##############################################################
    # 训练模型
    ckp_name = './checkpoint/{}_{}_iteration_{}_model.pth'.format(args.ckp_prefix, args.dataset, session)
    print('ckp_name', ckp_name)

    # 冻结部分模型参数，针对小样本持续学习
    if args.model == 'resnet20':
        if args.frozen_backbone_part and session > start_session:
            print('freeze part of the backbone')
            for name, param in tg_model.named_parameters():
                if name == 'conv1.weight' or name == 'bn1.weight' or name == 'bn1.bias':
                    param.requires_grad = False
                else:
                    if name[0:6] == 'layer1' or name[0:6] == 'layer2':
                        param.requires_grad = False
                    else:
                        print(name)
            tg_params = filter(lambda p: p.requires_grad, tg_model.parameters())
        else:
            tg_params = tg_model.parameters()
    elif args.model == 'resnet18':
        if args.frozen_backbone_part and session > start_session:
            print('freeze part of the backbone')
            for name, param in tg_model.named_parameters():
                if name == 'conv1.weight' or name == 'bn1.weight' or name == 'bn1.bias':
                    param.requires_grad = False
                else:
                    if name[0:6] == 'layer1' or name[0:6] == 'layer2' or name[0:6] == 'layer3':
                        param.requires_grad = False
                    else:
                        print(name)
            tg_params = filter(lambda p: p.requires_grad, tg_model.parameters())
        else:
            tg_params = tg_model.parameters()
    elif args.model == 'resnet32':
        tg_params = tg_model.parameters()
    else:
        raise ValueError('model {} not supported'.format(args.model))

    tg_model = tg_model.to(device)
    if session > start_session:
        ref_model = ref_model.to(device)
        print('the learning rate is {}'.format(base_lr))

    tg_optimizer = optim.SGD(tg_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
    if args.schedule == 'Milestone':
        if session > start_session:
            tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat_new, gamma=lr_factor)
        else:
            tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
    
    elif args.schedule == 'cosine':
        tg_lr_scheduler = utils_pytorch.get_cosine_schedule_with_warmup(tg_optimizer, num_training_steps=args.epochs*args.u_iter, num_warmup_steps=args.warmup_epochs*args.u_iter)
    else:
        tg_lr_scheduler = lr_scheduler.StepLR(tg_optimizer, step_size=lr_strat[0], gamma=lr_factor)
    
    print("iteration: {}, trainloader dataset size: {}, trainset size: {}".format(session, len(trainloader.dataset), len(trainset)))
    print("trainloader.dataset classes: {}".format(np.unique(trainloader.dataset.targets, return_counts=True)))
    print("trainset classes: {}".format(np.unique(trainset.targets, return_counts=True)))
    
    print("unlabels dataset size: {}".format(len(unlabeled_data) if unlabeled_data is not None else 0))
    if unlabeled_data is not None and len(unlabeled_data) > 0:
        print("unlabels dataset classes: {}".format(np.unique(unlabeled_gt, return_counts=True)))
    
    
    if args.use_class_weight and session > start_session:
        weight_per_base_class = [5.0 for _ in range((session) * args.nb_cl)]
        weight_per_novel_class = [1.0 for _ in range((session) * args.nb_cl, (session+1) * args.nb_cl)]
        weight_per_class = weight_per_base_class + weight_per_novel_class
        print("weight_per_class: {}".format(weight_per_class))
    else:
        weight_per_class = None

    prototypes_dict = {}
    print("Before training prototypes size: {}".format(len(prototypes)))
    for i in range(len(prototypes)):
        prototypes_dict["prototypes[{}]".format(i)] = len(prototypes[i])
    print("prototypes: {}".format(prototypes_dict))

    tg_model = incremental_train_and_eval(args=args, 
                                        base_lamda=args.base_lamda, 
                                        adapt_lamda=args.adapt_lamda, 
                                        u_t=args.u_t, 
                                        label2id=label2id, 
                                        uncertainty_distillation=args.uncertainty_distillation, 
                                        prototypes_list=prototypes, 
                                        prototypes_flag=prototypes_flag, 
                                        prototypes_on_flag=prototypes_on_flag, 
                                        update_unlabeled=args.update_unlabeled, 
                                        epochs=args.epochs, 
                                        method=args.method, 
                                        unlabeled_num=args.unlabeled_num, 
                                        unlabeled_iteration=args.unlabeled_iteration, 
                                        unlabeled_num_selected=args.unlabeled_num_selected, 
                                        train_batch_size=train_batch_size, 
                                        tg_model=tg_model, 
                                        ref_model=ref_model, 
                                        tg_optimizer=tg_optimizer, 
                                        tg_lr_scheduler=tg_lr_scheduler,
                                        trainloader=trainloader, 
                                        testloader=testloader,
                                        weight_per_class=None,
                                        iteration=session, 
                                        start_iteration=start_session,
                                        T=args.T, beta=args.beta, 
                                        unlabeled_data=unlabeled_data, 
                                        unlabeled_gt=unlabeled_gt, 
                                        nb_cl_fg=args.nb_cl_fg,
                                        nb_cl=args.nb_cl, 
                                        trainset=trainset, 
                                        image_size=args.image_size,
                                        text_anchor=text_anchor, 
                                        con_margin=args.con_margin,
                                        hard_negative=args.hard_negative,
                                        device=device,
                                        use_conloss=args.use_conloss,
                                        include_unlabel=args.include_unlabel,
                                        use_da=args.use_da,
                                        use_proto=args.use_proto,
                                        update_proto=args.update_proto,
                                        u_ratio=args.u_ratio,
                                        lambda_kd=args.lambda_kd,
                                        lambda_con=args.lambda_con, 
                                        lambda_cons=args.lambda_cons,
                                        lambda_reg=args.lambda_reg,
                                        lambda_in=args.lambda_in,
                                        lambda_cat=args.lambda_cat,
                                        lambda_ukd=args.lambda_ukd,
                                        use_proto_classifier=args.use_proto_classifer,
                                        u_iter=args.u_iter,
                                        no_use_conloss_on_ulb=args.no_use_conloss_on_ulb,
                                        unlabels_predict_mode=args.unlabels_predict_mode,
                                        use_sim=args.use_sim,
                                        use_lb_kd=args.use_lb_kd,
                                        use_srd=args.use_srd,
                                        use_session_labels=args.use_session_labels,
                                        p_cutoff=args.p_cutoff,
                                        warmup_epochs=args.warmup_epochs,
                                        use_feats_kd=args.use_feats_kd,
                                        use_ulb_aug=args.use_ulb_aug,
                                        q_cutoff=args.q_cutoff,
                                        adapt_weight=args.adapt_weight,
                                        use_mix_up=args.use_mix_up,
                                        mixup_alpha=args.mixup_alpha,
                                        use_hard_labels=args.use_hard_labels,
                                        use_old=args.use_old,
                                        use_metric_loss=args.use_metric_loss,
                                        kd_only_old=args.kd_only_old,
                                        kd_mode=args.kd_mode,
                                        use_ulb_kd=args.use_ulb_kd,
                                        ulb_kd_mode=args.ulb_kd_mode,
                                        dim=args.dim,
                                        use_adv=args.use_adv,
                                        proto_clissifier=args.proto_clissifier,
                                        ckp_prefix=args.ckp_prefix,
                                        is_fewshot=args.is_fewshot,)
    
    print("After training prototypes size: {}".format(len(prototypes)))
    for i in range(len(prototypes)):
        prototypes_dict["prototypes[{}]".format(i)] = len(prototypes[i])
    print("prototypes: {}".format(prototypes_dict))

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    
    torch.save(tg_model, ckp_name)
    
    if ssl_trainloader is not None and not args.no_use_filled:
        print('Filling buffer...')
        fill_pro_list(prototypes, tg_model, ssl_trainloader, device, args.k_shot, session * args.nb_cl)
        print("After filling prototypes size: {}".format(len(prototypes)))
        for i in range(len(prototypes)):
            prototypes_dict["prototypes[{}]".format(i)] = len(prototypes[i])
        print("prototypes: {}".format(prototypes_dict))
    print('Updating exemplar set...')
    
    dr_herding = []

    nb_protos_cl = args.buffer_size//((session+1)*args.nb_cl)
    print('nb_protos_cl: ', nb_protos_cl)
    if args.use_session_labels and session > start_session:
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-4])
    else:
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-3])
    
    num_features = tg_model.fc.in_features
    # 第一个session时, session=9, last_iter=0,所以不能用 session*args.nb_cl
    start_idx = last_iter * args.nb_cl
    end_idx = (session + 1) * args.nb_cl
    max_length = max(len(prototypes[i]) for i in range(start_idx, end_idx)) 
    
    for i in range(start_idx, end_idx):
        lst = prototypes[i]
        extended_list = list(lst) * (max_length // len(lst)) + list(lst)[:max_length % len(lst)]
        prototypes[i] = np.array(extended_list)
        lst = prototypes_flag[i]
        extended_list = list(lst) * (max_length // len(lst)) + list(lst)[:max_length % len(lst)]
        prototypes_flag[i] = np.array(extended_list)
        lst = prototypes_on_flag[i]
        extended_list = list(lst) * (max_length // len(lst)) + list(lst)[:max_length % len(lst)]
        prototypes_on_flag[i] = np.array(extended_list)

    for iter_dico in range(last_iter * args.nb_cl, (session + 1) * args.nb_cl):
        evalset = BaseDataset("test", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
        evalset.data = prototypes[iter_dico]
        evalset.targets = np.zeros(len(evalset))  # zero labels
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                 shuffle=False, num_workers=4)
        num_samples = len(evalset)
        mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features, device=device)
        D = mapped_prototypes.T
        D = D / np.linalg.norm(D, axis=0)

        herding = np.zeros(len(prototypes[iter_dico]), np.float32)
        dr_herding.append(herding)
        # Herding procedure : ranking of the potential exemplars
        mu = np.mean(D, axis=1)
        index1 = int(iter_dico / args.nb_cl)
        index2 = iter_dico % args.nb_cl
        dr_herding[index2] = dr_herding[index2] * 0
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not (np.sum(dr_herding[index2] != 0) == min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if dr_herding[index2][ind_max] == 0:
                dr_herding[index2][ind_max] = 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]

        if (iter_dico + 1) % args.nb_cl == 0:
            alpha_dr_herding.append(np.array(dr_herding))
            dr_herding = []

    X_protoset_cumuls = []
    Y_protoset_cumuls = []
    X_protoset_cumuls_flag = []
    X_protoset_cumuls_on_flag = []


    # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
    print('Computing mean-of-exemplars...')
    class_means = np.zeros((args.proto_dim, args.num_classes, 3))

    for iteration2 in range(session+1):
        for iter_dico in range(args.nb_cl):
            current_cl = order[range(iteration2*args.nb_cl, (iteration2+1)*args.nb_cl)]

            # Collect data in the feature space for each class
            evalset = BaseDataset("test", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
            evalset.data = prototypes[iteration2*args.nb_cl+iter_dico]
            evalset.targets = np.zeros(evalset.data.shape[0]) #zero labels
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=4)
            num_samples = evalset.data.shape[0]
            mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features, device=device)
            D = mapped_prototypes.T
            D = D/np.linalg.norm(D,axis=0)
            
            # Flipped version also
            evalset.data = prototypes[iteration2*args.nb_cl+iter_dico]
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=4)
            mapped_prototypes2 = compute_features(tg_feature_model, evalloader, num_samples, num_features,device=device)
            D2 = mapped_prototypes2.T
            D2 = D2/np.linalg.norm(D2,axis=0)
            

            # iCaRL 计算准确度使用以及保存样本使用
            alph = alpha_dr_herding[iteration2][iter_dico]
            alph = (alph>0)*(alph<nb_protos_cl+1)*1.
            
            if args.adapt_filled:  
                lst = prototypes[iteration2*args.nb_cl+iter_dico]
                extended_list_p = list(lst) * (nb_protos_cl // len(lst)) + list(lst)[:nb_protos_cl % len(lst)]
                lst = prototypes_flag[iteration2*args.nb_cl+iter_dico]
                extended_list_pf = list(lst) * (nb_protos_cl // len(lst)) + list(lst)[:nb_protos_cl % len(lst)]
                lst = prototypes_on_flag[iteration2*args.nb_cl+iter_dico]
                extended_list_pof = list(lst) * (nb_protos_cl // len(lst)) + list(lst)[:nb_protos_cl % len(lst)]

                X_protoset_cumuls.append(np.array(extended_list_p))
                X_protoset_cumuls_flag.append(np.array(extended_list_pf))
                X_protoset_cumuls_on_flag.append(np.array(extended_list_pof))
                Y_protoset_cumuls.append(order[iteration2*args.nb_cl+iter_dico]*np.ones(nb_protos_cl))
            else:  
                X_protoset_cumuls.append(prototypes[iteration2*args.nb_cl+iter_dico][np.where(alph==1)[0]])
                X_protoset_cumuls_flag.append(prototypes_flag[iteration2 * args.nb_cl + iter_dico][np.where(alph == 1)[0]])
                X_protoset_cumuls_on_flag.append(prototypes_on_flag[iteration2 * args.nb_cl + iter_dico][np.where(alph == 1)[0]])
                Y_protoset_cumuls.append(order[iteration2*args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
            print('X_protoset_cumuls:', len(X_protoset_cumuls), len(X_protoset_cumuls[0]))
            alph = alph/np.sum(alph)
            print(D.shape, alph.shape)
            class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
            class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])

            # Normal NCM
            alph = np.ones(len(prototypes[iteration2*args.nb_cl+iter_dico])) / len(prototypes[iteration2*args.nb_cl+iter_dico])
            
            class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
            class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])

            # dividing labeled and unlabeled and compute class-means
            alph = np.zeros(len(prototypes[iteration2*args.nb_cl+iter_dico]))
            num_labeled = np.sum(prototypes_flag[iteration2*args.nb_cl+iter_dico], axis=0)
            num_unlabeled = len(prototypes[iteration2*args.nb_cl+iter_dico]) - num_labeled
            alph_labeled = 2 / (2 * num_labeled + num_unlabeled)
            alph_unlabeled = 1 / (2 * num_labeled + num_unlabeled)
            for i in range(len(prototypes[iteration2*args.nb_cl+iter_dico])):
                if prototypes_flag == 1:
                    alph[i] = alph_labeled
                else:
                    alph[i] = alph_unlabeled

            class_means[:, current_cl[iter_dico], 2] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
            class_means[:, current_cl[iter_dico], 2] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 0])

    if not os.path.isdir('checkpoint'):
        torch.save(class_means, './checkpoint/{}_run_iteration_{}_class_means.pth'.format(args.ckp_prefix, session))

    current_means = class_means[:, order[range(0, (session+1)*args.nb_cl)]]
    
    if args.use_session_means:
        print('Computing mean-of-session...')
        session_means = np.zeros((args.dim, session - start_session + 1))
        print('session_means shape: {}'.format(session_means.T.shape))
        for cur_session in range(start_session, session + 1):
            evalset = BaseDataset("test", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
            if cur_session == start_session:
                evalset.data = np.concatenate(prototypes[0 * args.nb_cl: (cur_session + 1) * args.nb_cl])
            else:
                evalset.data = np.concatenate(prototypes[cur_session * args.nb_cl: (cur_session + 1) * args.nb_cl])
                evalset.data = np.concatenate((evalset.data, unlabeled_data))
            evalset.targets = np.zeros(evalset.data.shape[0]) #zero labels
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=4)
            num_samples = evalset.data.shape[0]
            if cur_session == start_session:
                print("For session: {}, evalset size: {}, include class: {}".format(cur_session, num_samples, [i for i in range(0 * args.nb_cl, (cur_session + 1) * args.nb_cl)]))
            else:
                print("For session: {}, evalset size: {}, include class: {}".format(cur_session, num_samples, [i for i in range(cur_session * args.nb_cl, (cur_session + 1) * args.nb_cl)]))
            mapped_prototypes = compute_feats(tg_model, evalloader, num_samples, args.dim, device=device)
            D3 = mapped_prototypes.T
            D3 = D3/np.linalg.norm(D3,axis=0)
            session_means[:, cur_session - start_session] = np.mean(D3, axis=1)
    else:
        session_means = None

    print('Computing last cumulative accuracy...')
    evalset = BaseDataset("test", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
    evalset.data = X_valid_cumul
    evalset.targets = Y_valid_cumul
    print('evalset size: {}, trans: {}'.format(len(evalset), evalset.transform))
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=4)
    cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, 
                                text_anchor, print_info=True, session_means=session_means, 
                                start_session=start_session, nb_cl=args.nb_cl, device=device)

    # 评估模型性能
    if session > start_session:
        print('Computing last accuracy of base classes...')
        evalset = BaseDataset("test", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
        evalset.data = X_valid_cumul_base
        evalset.targets = Y_valid_cumul_base
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=4)
        cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, text_anchor, device=device)
        print('Computing last accuracy of novel classes...')
        evalset = BaseDataset("test", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
        evalset.data = X_valid_cumul_novel
        evalset.targets = Y_valid_cumul_novel
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                 shuffle=False, num_workers=4)
        cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, text_anchor, device=device)

    if args.resume and session == start_session:
        pass
    else:
        print('Computing best cumulative accuracy...')
        eval_tg_model = torch.load('./checkpoint/{}_best_model_session_{}.pth'.format(args.ckp_prefix, session))
        if args.use_session_labels and session > start_session:
            tg_feature_model = nn.Sequential(*list(eval_tg_model.children())[:-4])
        else:
            tg_feature_model = nn.Sequential(*list(eval_tg_model.children())[:-3])
        evalset = BaseDataset("test", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
        evalset.data = X_valid_cumul
        evalset.targets = Y_valid_cumul
        print('evalset size: {}, trans: {}'.format(len(evalset), evalset.transform))
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=4)
        cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, 
                                text_anchor, print_info=True, session_means=session_means, 
                                start_session=start_session, nb_cl=args.nb_cl, device=device)
        # 评估模型性能
        if session > start_session:
            print('Computing best accuracy of base classes...')
            evalset = BaseDataset("test", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
            evalset.data = X_valid_cumul_base
            evalset.targets = Y_valid_cumul_base
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=4)
            cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, text_anchor, device=device)
            print('Computing best accuracy of novel classes...')
            evalset = BaseDataset("test", args.image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
            evalset.data = X_valid_cumul_novel
            evalset.targets = Y_valid_cumul_novel
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                    shuffle=False, num_workers=4)
            cumul_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader, text_anchor, device=device)
