#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
import random
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pdb
import math
from torch.utils.data import BatchSampler, RandomSampler
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter

from utils_pytorch import *
from .dist_align import DistAlignQueueHook
from .cat_kd import CAT_KD
from .AT import AT
from .attack import Attack
from dataloder import BaseDataset, UnlabelDataset, ReservedUnlabelDataset
from utils_incremental.compute_accuracy import compute_accuracy_train
def incremental_train_and_eval(args, base_lamda, adapt_lamda, u_t, label2id, uncertainty_distillation, 
                               prototypes_list, prototypes_flag, prototypes_on_flag, update_unlabeled, 
                               epochs, method, unlabeled_num, unlabeled_iteration, unlabeled_num_selected, 
                               train_batch_size, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, 
                               testloader, iteration, start_iteration, T, beta, unlabeled_data, unlabeled_gt, nb_cl_fg, 
                               nb_cl, trainset, image_size, text_anchor, use_conloss=True, include_unlabel=True,
                               con_margin=0.2, hard_negative=False, fix_bn=False, weight_per_class=None, 
                               device=None, use_da=False, use_proto=False, update_proto=False, u_ratio=1, lambda_kd=1.0, lambda_mixup=1.0,
                               lambda_con=1.0, lambda_cons=1.0, lambda_in=1.0, lambda_reg=1.0, lambda_session=1.0, lambda_cat=10.0, lambda_ce=1.0,
                               use_proto_classifier=False, lambda_metric=1.0, lambda_ukd=1.0, kd_only_old=False, u_iter=100, no_use_conloss_on_ulb=False, 
                               unlabels_predict_mode='sqeuclidean',use_sim=False, smoothing_alpha=0.7, p_cutoff=0.0, q_cutoff=0.25, 
                               use_ulb_kd=False, use_lb_kd=False, use_srd=False, use_session_labels=False, lambda_proto=1.0,
                               warmup_epochs=100, dim=512, use_feats_kd=False, use_ulb_aug=False, adapt_weight=False, use_mix_up=False, 
                               mixup_alpha=0.75, use_hard_labels=True, use_old=True, use_metric_loss=False, kd_mode='logits', ulb_kd_mode='logits',
                               use_adv=False, lambda_adv=0.1, adv_num=200, adv_epochs=3, adv_alpha=25, proto_clissifier=False,me_max=True,cm=None,ckp_prefix='',
                               is_fewshot=False):

    N = 128
    
    ema_bank = 0.1
    smoothing_alpha = 0.9
    use_ema_teacher = False
    mem_bank = torch.randn(dim, len(trainset)).to(device)
    mem_bank = F.normalize(mem_bank, dim=0)
    labels_bank = torch.zeros(len(trainset), dtype=torch.long).to(device)
    mem_bank, labels_bank = mem_bank.detach(), labels_bank.detach()

    ref_mem_bank = torch.randn(dim, len(trainset)).to(device)
    ref_mem_bank = F.normalize(ref_mem_bank, dim=0)
    ref_labels_bank = torch.zeros(len(trainset), dtype=torch.long).to(device)
    ref_mem_bank, ref_labels_bank = ref_mem_bank.detach(), ref_labels_bank.detach()
    
    def update_bank(k, labels, index):
        mem_bank[:, index] = F.normalize(k).t().detach()
        labels_bank[index] = labels.detach()

    def update_ref_bank(k, labels, index):
        ref_mem_bank[:, index] = F.normalize(k).t().detach()
        ref_labels_bank[index] = labels.detach()
    
    old_cn = iteration * nb_cl
    total_cn = (iteration + 1) * nb_cl
    
    if old_cn == 0:
        prototypes_old = torch.tensor([]).to(device)
    else:
        prototypes_old = torch.randn(old_cn, dim).to(device)
    prototypes_new = torch.randn(nb_cl, dim).to(device)

    if old_cn == 0:
        prototypes_ref_old = torch.tensor([]).to(device)
    else:
        prototypes_ref_old = torch.randn(old_cn, dim).to(device)
    prototypes_ref_new = torch.randn(nb_cl, dim).to(device)

    writer = SummaryWriter(log_dir='checkpoint/logs/{}/{}'.format(args.ckp_prefix, iteration))
    # 参考USB里的DA实现
    distri = DistAlignQueueHook(num_classes=nb_cl, queue_length=N, p_target_type='uniform')
    
    if is_fewshot:
        if iteration == start_iteration:
            include_unlabel = False

    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features

        assert num_old_classes == old_cn     
        prototypes_ref_old, prototypes_ref_new, prototypes_ref = get_proto(trainloader, ref_model, old_cn, device, False)

    if use_conloss:
        text_anchor = text_anchor.to(device)
    
    if use_metric_loss:
        triplet_loss = SupConLoss(temperature=0.1, contrast_mode='all', base_temperature=0.1)
    
    if include_unlabel:
        unlabeled_trainset = UnlabelDataset(image_size, dataset=args.dataset)
        unlabeled_trainset.data = unlabeled_data
        unlabeled_trainset.targets = unlabeled_gt
        ssl_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, 
                                                    batch_size=u_ratio*train_batch_size, 
                                                    shuffle=True, num_workers=4) 
        ssl_iterator = iter(ssl_trainloader)  

    best_acc = 0
    prototypes_old, prototypes_new, pro = get_proto(trainloader, tg_model, old_cn, device, False)
    # train the model with labeled data
    for epoch in range(epochs):
        # 配合proto_clissifier的权重设置
        # lambda_ce, lambda_cons = 0.001, 0.001
        # lambda_con = epoch / epochs
        # train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        total = 0
        correct = 0
        ulb_total = 0
        ulb_correct = 0
        ulb_mask_total = 0
        ulb_mask_correct = 0
        train_loss = 0
        train_suploss_kd = 0
        train_suploss_adv = 0
        train_suploss_feats_kd = 0
        train_suploss_lb = 0
        train_conloss_lb = 0
        train_metric_loss_lb = 0
        train_metric_loss_ulb = 0
        train_conloss_ulb = 0
        train_consloss_ulb = 0
        train_consloss_ulb_aug = 0
        train_suploss_kd_ulb = 0
        train_suploss_proto = 0
        train_suploss_proto_ulb = 0
        train_inloss_ulb = 0
        train_rloss_ulb = 0
        train_util_ratio = 0
        train_n_util_ratio = 0
        train_q_util_ratio = 0
        train_mixup_loss = 0
        mean_pseudo_label = []
        x_min, x_max = None, None
                        
        if epoch % 10 == 0:
            print('\nEpoch: %d, LR: ' % epoch, end='')
            print(tg_lr_scheduler.get_last_lr())
        
        for batch_idx, (indexs, inputs, inputs_s, targets, flags, on_flags) in enumerate(trainloader):
            tg_optimizer.zero_grad()
            indexs, inputs, inputs_s, targets, flags, on_flags = indexs.to(device), inputs.to(device), inputs_s.to(device), targets.to(device), flags.to(device), on_flags.to(device)
            
            if batch_idx == 0:
                x_min, x_max = inputs.min(), inputs.max()   
            else:
                x_min, x_max = min(x_min, inputs.min()), max(x_max, inputs.max())

            num_lb = len(targets)
            if num_lb == 1:
                continue
            
            outputs, raw_feats, feats, session_outputs = tg_model(inputs, return_feats_list=True)
            outputs_s, raw_feats_s, feats_s, session_outputs_s = tg_model(inputs_s, return_feats=True)
            update_bank(feats, targets, indexs)
            
            suploss_lb = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())
            
            # 将提取的视觉特征与text特征空间对齐
            if use_conloss:    
                # COSINE的另一种实现方式
                scores = F.linear(F.normalize(feats, p=2, dim=1), F.normalize(text_anchor, p=2, dim=1)) / 0.1
                conloss_lb = F.cross_entropy(scores, targets.long())
            else:
                conloss_lb = torch.tensor(0.0).to(device)
            
            if use_metric_loss:
                metric_loss_lb = triplet_loss(torch.stack([F.normalize(feats, p=2, dim=1), 
                                                           F.normalize(feats_s, p=2, dim=1)], dim=1), 
                                              targets, device=device)
            else:
                metric_loss_lb = torch.tensor(0.0).to(device)

            if proto_clissifier:
                prototypes = torch.cat([prototypes_old, prototypes_new], dim=0)
                # COSINE的另一种实现方式
                outputs_proto = F.linear(F.normalize(feats, p=2, dim=1), F.normalize(prototypes, p=2, dim=1)) / 0.1
                suploss_proto = nn.CrossEntropyLoss(weight_per_class)(outputs_proto, targets.long())
            else:
                suploss_proto = torch.tensor(0.0).to(device)
                
            if iteration > start_iteration:
                ref_outputs, ref_raw_feats, ref_feats, ref_session_outputs= ref_model(inputs, return_feats_list=True)
                update_ref_bank(ref_feats, targets, indexs)
                old_mask = targets < num_old_classes

                if kd_mode == 'logits':
                    if kd_only_old:
                        if old_mask.sum() > 0:
                            suploss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[old_mask][:, :num_old_classes] / T, dim=1),
                                            F.softmax(ref_outputs[old_mask].detach() / T, dim=1)) * T * T * beta * num_old_classes
                        else:
                            suploss_kd = torch.tensor(0.0).to(device)
                    else:
                        suploss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
                                        F.softmax(ref_outputs.detach() / T, dim=1)) * T * T * beta * num_old_classes
                    suploss_feats_kd = torch.tensor(0.0).to(device)

                elif kd_mode == 'feats':
                    if kd_only_old:
                        if old_mask.sum() > 0:
                            suploss_kd = F.mse_loss(feats[old_mask], ref_feats[old_mask].detach()) * 1e3
                        else:
                            suploss_kd = torch.tensor(0.0).to(device)
                    else:
                        suploss_kd = F.mse_loss(feats, ref_feats.detach())  * 1e3
                    suploss_feats_kd = torch.tensor(0.0).to(device)

                else:
                    raise ValueError('kd_mode: {} not supported'.format(kd_mode))  
            else:
                suploss_kd = torch.tensor(0.0).to(device)
                suploss_feats_kd = torch.tensor(0.0).to(device)

            skip = False
            if include_unlabel and epoch >= warmup_epochs:                           
                try:
                    inputs_ulb, inputs_s_ulb, gt = next(ssl_iterator)
                except StopIteration:
                    ssl_iterator = iter(ssl_trainloader)
                    inputs_ulb, inputs_s_ulb, gt = next(ssl_iterator)
                
                num_ulb = len(gt)
                if num_ulb == 1:
                    skip = True
                    continue

                inputs_ulb, inputs_s_ulb, gt = inputs_ulb.to(device), inputs_s_ulb.to(device), gt.to(device)
                
                outputs_ulb, raw_feats_ulb, feats_ulb, session_outputs_ulb = tg_model(inputs_ulb, return_feats=True)
                outputs_s_ulb, raw_feats_s_ulb, feats_s_ulb, session_outputs_s_ulb = tg_model(inputs_s_ulb, return_feats=True)
                feats_ulb, feats_s_ulb = F.normalize(feats_ulb, p=2, dim=1), F.normalize(feats_s_ulb, p=2, dim=1)

                if use_proto:
                    if unlabels_predict_mode == 'cosine':
                        # 只使用新类原型打伪标签
                        cosine_scores = F.linear(feats_ulb, F.normalize(prototypes_new, p=2, dim=1)) / 0.1 # COSINE的另一种实现方式
                        pseudo_label = torch.softmax(cosine_scores, dim=1)
                        # DA
                        pseudo_label = distri.dist_align(probs_x_ulb=pseudo_label.detach())
                        max_probs, predicted_classes = torch.max(pseudo_label, dim=1)
                        mask = max_probs.ge(p_cutoff).float()
                        n_mask = max_probs.le(q_cutoff).float()
                    elif unlabels_predict_mode == 'sqeuclidean':
                        class_means_squared = torch.sum(F.normalize(prototypes_new, p=2, dim=1)**2, dim=1, keepdim=True)  # (num_classes, 1)
                        outputs_feature_squared = torch.sum(feats_ulb**2, dim=1, keepdim=True).T  # (1, batch_size)
                        dot_product = torch.matmul(F.normalize(prototypes_new, p=2, dim=1), feats_ulb.T)  # (num_classes, batch_size)
                        squared_distances = class_means_squared + outputs_feature_squared - 2 * dot_product  # (num_classes, batch_size)
                        pseudo_label = torch.softmax(-torch.sqrt(squared_distances.T), dim=1)  # (num_classes, batch_size)
                        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
                        mask = max_probs.ge(p_cutoff).float()
                        n_mask = max_probs.le(q_cutoff).float()
                        predicted_classes = torch.argmin(squared_distances, dim=0)  # (batch_size,)
                    else:
                        raise ValueError('unlabels_predict_mode: {} not supported'.format(unlabels_predict_mode))
                    rloss = 0.0

                elif use_sim:
                    bank = mem_bank.clone().detach()
                    with torch.no_grad():
                        # 先验屏蔽掉旧类
                        if old_cn > 0:
                            outputs_ulb[:, :old_cn] = -1e4
                        probs_x_ulb_w = F.softmax(outputs_ulb, dim=-1)
                        teacher_logits = feats_ulb @ bank
                        if old_cn > 0:
                            teacher_logits[:, labels_bank<old_cn] = -1e4
                        teacher_prob_orig = F.softmax(teacher_logits / T, dim=1)

                        factor = probs_x_ulb_w.gather(1, labels_bank.expand([num_ulb, -1]))
                        teacher_prob = teacher_prob_orig * factor
                        teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                        if smoothing_alpha < 1:
                            bs = teacher_prob_orig.size(0)
                            aggregated_prob = torch.zeros([bs, total_cn], device=teacher_prob_orig.device)
                            aggregated_prob = aggregated_prob.scatter_add(1, labels_bank.expand([bs,-1]) , teacher_prob_orig)
                            pseudo_label = probs_x_ulb_w * smoothing_alpha + aggregated_prob * (1-smoothing_alpha)
                        else:
                            pseudo_label = probs_x_ulb_w

                    student_logits = feats_s_ulb @ bank
                    student_prob = F.softmax(student_logits / T, dim=1)

                    in_loss = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1).mean()
                    
                    sharp_p = student_prob / torch.sum(student_prob, dim=1, keepdim=True)
                    rloss = 0.0
                    if me_max:
                        avg_probs = torch.mean(sharp_p, dim=0)
                        rloss -= torch.sum(torch.log(avg_probs**(-avg_probs)))
                    
                    if epoch == 0:
                        in_loss = 0.0
                        rloss = 0.0
                        pseudo_label = probs_x_ulb_w
                    
                    max_probs, predicted_classes = torch.max(pseudo_label, dim=1)
                    mask = max_probs.ge(p_cutoff).float()
                    n_mask = max_probs.le(q_cutoff).float()

                else:
                    pseudo_label = torch.softmax(outputs_ulb[:, old_cn:total_cn], dim=-1)
                    # DA
                    pseudo_label = distri.dist_align(probs_x_ulb=pseudo_label.detach())
                    max_probs, predicted_classes = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(p_cutoff).float()
                    n_mask = max_probs.le(q_cutoff).float()
                    rloss = 0.0

                # 伪标签打法 
                if use_sim:
                    consloss_ulb = ce_loss(outputs_s_ulb, pseudo_label, False, reduction='none') * mask 
                
                elif use_proto:
                    if use_hard_labels:
                        consloss_ulb = ce_loss(outputs_s_ulb, predicted_classes, True, reduction='none') * mask
                    else:
                        pseudo_label = pseudo_label**2 / torch.sum(pseudo_label**2, dim=1, keepdim=True)
                        consloss_ulb = ce_loss(outputs_s_ulb, pseudo_label, False, reduction='none') * mask
                    in_loss = torch.tensor(0.0).to(device)

                else:
                    mean_pseudo_label.append(pseudo_label.mean(0))
                    predicted_classes = predicted_classes + old_cn
                    consloss_ulb = ce_loss(outputs_s_ulb, predicted_classes, True, reduction='none') * mask
                    in_loss = torch.tensor(0.0).to(device)

                consloss_ulb = consloss_ulb.mean()

                if proto_clissifier:
                    prototypes = torch.cat([prototypes_old, prototypes_new], dim=0) 
                    # COSINE的另一种实现方式
                    outputs_proto_ulb = F.linear(feats_s_ulb, F.normalize(prototypes, p=2, dim=1)) / 0.1
                    if use_hard_labels:
                        suploss_proto_ulb = ce_loss(outputs_proto_ulb, predicted_classes, True, reduction='none') * mask
                    else:
                        pseudo_label = pseudo_label**2 / torch.sum(pseudo_label**2, dim=1, keepdim=True)
                        suploss_proto_ulb = ce_loss(outputs_proto_ulb, pseudo_label, False, reduction='none') * mask
                    suploss_proto_ulb = suploss_proto_ulb.mean()
                else:
                    suploss_proto_ulb = torch.tensor(0.0).to(device)

                ulb_total += gt.size(0)
                ulb_correct += predicted_classes.eq(gt).sum().item()
                pseudo_acc = predicted_classes.eq(gt).sum().item() / gt.size(0)
                
                if mask.bool().any():
                    ulb_mask_total += gt[mask.bool()].size(0)
                    ulb_mask_correct += predicted_classes[mask.bool()].eq(gt[mask.bool()]).sum().item()
                    mask_pseudo_acc = predicted_classes[mask.bool()].eq(gt[mask.bool()]).sum().item() / gt[mask.bool()].size(0)

                if not mask.bool().all():
                    no_mask_pseudo_acc = predicted_classes[torch.logical_not(mask.bool())].eq(gt[torch.logical_not(mask.bool())]).float().mean().item()
                
                # 对比学习损失
                if use_metric_loss:
                    metric_loss_ulb = triplet_loss(torch.stack([feats_ulb, feats_s_ulb], dim=1), device=device)
                else:
                    metric_loss_ulb = torch.tensor(0.0).to(device)
                
                if not no_use_conloss_on_ulb:
                    # COSINE的另一种实现方式
                    scores = F.linear(feats_ulb, F.normalize(text_anchor, p=2, dim=1)) / 0.1
                    
                    conloss_ulb = F.cross_entropy(scores, predicted_classes.long(), reduction='none') * mask
                    conloss_ulb = conloss_ulb.mean()
                else:
                    conloss_ulb = torch.tensor(0.0).to(device)
                
                # 无标记数据蒸馏
                if iteration > start_iteration and use_ulb_kd:

                    if ulb_kd_mode == 'logits':
                        ref_outputs_ulb = ref_model(inputs_ulb)
                        ref_predicted_classes = ref_outputs_ulb.max(1)[1].reshape(-1)
                        
                        gt_mask = torch.zeros_like(ref_outputs_ulb).scatter_(1, ref_predicted_classes.unsqueeze(1), 1).bool()
                        pred_teacher_part2 = F.softmax(ref_outputs_ulb / T - 1000.0 * gt_mask, dim=1)
                        log_pred_student_part2 = F.log_softmax(outputs_ulb[:, :num_old_classes] / T - 1000.0 * gt_mask, dim=1)
                        
                        suploss_kd_ulb = (
                            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
                            * (T**2)
                            / num_ulb
                        )
                        
                    elif ulb_kd_mode == 'feats':
                        ref_outputs_ulb, ref_raw_feats_ulb, ref_feats_ulb, _ = ref_model(inputs_ulb, return_feats=True)
                        suploss_kd_ulb = F.mse_loss(raw_feats_ulb, ref_raw_feats_ulb.detach())
                    
                    elif ulb_kd_mode == 'attention':
                        ref_outputs_ulb, ref_raw_feats_ulb, ref_feats_ulb, ref_session_outputs_ulb = ref_model(inputs_ulb, return_feats=True)
                        suploss_kd_ulb = AttLoss.forward_train(outputs_ulb, session_outputs_ulb, ref_outputs_ulb, ref_session_outputs_ulb)

                    elif ulb_kd_mode == 'cosine':
                        ref_outputs_ulb, ref_raw_feats_ulb, ref_feats_ulb, _ = ref_model(inputs_ulb, return_feats=True)
                        
                        normalized_ref_feats_ulb = F.normalize(ref_feats_ulb, p=2, dim=1)

                        scores_ref = F.cosine_similarity(prototypes_ref_old.unsqueeze(0).repeat(len(normalized_ref_feats_ulb), 1, 1),
                                                    normalized_ref_feats_ulb.unsqueeze(1).repeat(1, len(prototypes_ref_old), 1), 2) / 0.1
                        scores_tg = F.cosine_similarity(prototypes_old.unsqueeze(0).repeat(len(feats_ulb), 1, 1),
                                                    feats_ulb.unsqueeze(1).repeat(1, len(prototypes_old), 1), 2) / 0.1
                        
                        ref_predicted_classes = scores_ref.max(1)[1].reshape(-1)

                        gt_mask = torch.zeros_like(scores_ref).scatter_(1, ref_predicted_classes.unsqueeze(1), 1).bool()
                        pred_teacher_part2 = F.softmax(scores_ref - 1000.0 * gt_mask, dim=1)
                        log_pred_student_part2 = F.log_softmax(scores_tg  - 1000.0 * gt_mask, dim=1)
            
                        suploss_kd_ulb = (
                            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
                            * (0.1**2)
                            / num_ulb
                        )
                        
                    elif ulb_kd_mode == 'similarity':
                        # 仿照ICCV蒸馏的无标记数据蒸馏实现
                        _, _, ref_feats_ulb, _ = ref_model(inputs_s_ulb, return_feats=True)
                        
                        normalized_ref_feats_ulb = F.normalize(torch.cat((ref_feats,ref_feats_ulb)), p=2, dim=1)
                        
                        # 使用当前batch的旧类标记数据 --> similarity_part
                        if old_mask.sum() > 0:

                            prototypes_ref = F.normalize(prototypes_ref, p=2, dim=1)
                            num_prototypes = prototypes_ref.shape[0]
                            prototype_targets = torch.arange(num_prototypes, device=prototypes_ref.device)
                            labels_metric = F.one_hot(prototype_targets, num_classes=num_prototypes)

                            teacher_logits = normalized_ref_feats_ulb @ prototypes_ref.T
                            teacher_prob = F.softmax(teacher_logits / 0.1, dim=1)                  
                            student_logits = F.normalize(torch.cat((feats, feats_ulb)), p=2, dim=1) @ prototypes_ref.T
                            student_prob = F.log_softmax(student_logits / 0.1, dim=1)
                            
                            assert teacher_prob.size() == student_prob.size() 
                            suploss_kd_ulb = torch.sum(-teacher_prob.detach() * student_prob, dim=1).mean() * 1 #* 0.2
                        else:
                            suploss_kd_ulb = torch.tensor(0.0).to(device)
                    
                    else:
                        raise ValueError('ulb_kd_mode: {} not supported'.format(ulb_kd_mode))
                    
                    if adapt_weight:
                        suploss_kd_ulb = suploss_kd_ulb * (old_cn//(total_cn-old_cn))
                else:
                    suploss_kd_ulb = torch.tensor(0.0).to(device)

                # 使用原型给低置信度样本打旧类及新类伪标签
                if iteration > start_iteration and use_ulb_aug and epoch != 0:
                    prototypes = torch.cat([prototypes_old, prototypes_new], dim=0)
                    # COSINE的另一种实现方式
                    q_cosine_scores = F.linear(feats_ulb, F.normalize(prototypes, p=2, dim=1)) / 0.1

                    q_pseudo_label = torch.softmax(q_cosine_scores, dim=1)
                    q_predict_class = q_pseudo_label.max(1)[1]
                    
                    if not mask.bool().all():
                        ulb_aug_old_mask = torch.logical_and(q_predict_class.lt(old_cn), torch.logical_not(mask.bool()))
                        ulb_aug_new_mask = torch.logical_and(q_predict_class.ge(old_cn), torch.logical_not(mask.bool()))

                        old_class_num = ulb_aug_old_mask.sum().item()
                        new_class_num = ulb_aug_new_mask.sum().item()

                        if ulb_aug_new_mask.any():
                            ulb_aug_new_acc = q_predict_class[ulb_aug_new_mask].eq(gt[ulb_aug_new_mask]).float().mean().item()
                            ulb_new_acc = predicted_classes[ulb_aug_new_mask].eq(gt[ulb_aug_new_mask]).float().mean().item()
                        else:
                            ulb_aug_new_acc = torch.tensor(0.0).to(device)
                            ulb_new_acc = torch.tensor(0.0).to(device)
                    
                    consloss_ulb_aug = ce_loss(outputs_s_ulb, q_predict_class, True, reduction='none') * torch.logical_not(mask.bool()).float()
                                        
                    consloss_ulb_aug = consloss_ulb_aug.mean()
                else:
                    consloss_ulb_aug = torch.tensor(0.0).to(device)
                
                # mixup
                if iteration > start_iteration and use_mix_up and not mask.bool().all():
                    
                    mixup_x, mixup_u = inputs[targets<old_cn], inputs_ulb[torch.logical_not(mask.bool())]
                    q_pseudo_label = torch.softmax(outputs_ulb[torch.logical_not(mask.bool())], dim=-1)

                    mixup_y, mixup_p = F.one_hot(targets[targets<old_cn], total_cn), q_pseudo_label                                            
                    max_length = max(len(mixup_x), len(mixup_u))
                    
                    if len(mixup_x) < max_length:
                        # 计算重复次数和剩余长度
                        repeat_times = max_length // mixup_x.size(0)
                        remaining_length = max_length % mixup_x.size(0)

                        # 通过重复和切片来扩展 mixup_x
                        expanded_mixup_x = mixup_x.repeat(repeat_times, 1, 1, 1)
                        expanded_mixup_x = torch.cat((expanded_mixup_x, mixup_x[:remaining_length]), dim=0)

                        # 通过重复和切片来扩展 mixup_y
                        expanded_mixup_y = mixup_y.repeat(repeat_times, 1)
                        expanded_mixup_y = torch.cat((expanded_mixup_y, mixup_y[:remaining_length]), dim=0)

                        mixup_x, mixup_y = expanded_mixup_x, expanded_mixup_y
                    elif len(mixup_u) < max_length:
                        repeat_times = max_length // mixup_u.size(0)
                        remaining_length = max_length % mixup_u.size(0)

                        expanded_mixup_u = mixup_u.repeat(repeat_times, 1, 1, 1)
                        expanded_mixup_u = torch.cat((expanded_mixup_u, mixup_u[:remaining_length]), dim=0)

                        expanded_mixup_p = mixup_p.repeat(repeat_times, 1)
                        expanded_mixup_p = torch.cat((expanded_mixup_p, mixup_p[:remaining_length]), dim=0)

                        mixup_u, mixup_p = expanded_mixup_u, expanded_mixup_p
                    else:
                        pass

                    assert mixup_x.size() == mixup_u.size()
                    assert mixup_y.size() == mixup_p.size()

                    mixup_inputs = torch.cat([mixup_x, mixup_u], dim=0)
                    mixup_inputs_labels = torch.cat([mixup_y, mixup_p], dim=0)
                    mixed_x, mixed_y, lam = mixup_one_target(mixup_inputs, mixup_inputs_labels, 
                                                            mixup_alpha, is_bias=True)
                    mixup_outputs = tg_model(mixed_x)
                    ref_mixup_outputs = ref_model(mixed_x)
                    mixup_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(mixup_outputs[:, :num_old_classes] / T, dim=1),
                                                                     F.softmax(ref_mixup_outputs.detach() / T, dim=1)) * T * T * beta * num_old_classes
                else:
                    mixup_loss = torch.tensor(0.0).to(device)
                
                # 对抗训练增强
                if iteration > start_iteration and use_adv and mask.bool().any():
                    _, _, ref_feats_ulb, _ = ref_model(inputs_ulb, return_feats=True)
                    adv_x = inputs_ulb[mask.bool()]
                    ref_feats = ref_feats_ulb[mask.bool()]

                    if len(adv_x) >= adv_num:
                        
                        x, y = gen_adv_data(ref_model, adv_x, ref_feats, gt[mask.bool()], old_cn, prototypes_ref_old, 
                                            device, adv_num, adv_alpha, adv_epochs, x_min, x_max)
                        adv_data_num = len(x)
                        if adv_data_num > 0:
                            adv_outputs = tg_model(x)
                            suploss_adv = nn.CrossEntropyLoss(weight_per_class)(adv_outputs, y.long())
                        else:
                            suploss_adv = torch.tensor(0.0).to(device)
                    else:
                        adv_data_num = 0
                        suploss_adv = torch.tensor(0.0).to(device)
                else:
                    suploss_adv = torch.tensor(0.0).to(device)
                
                # 计算总损失
                loss = lambda_ce * suploss_lb \
                    + lambda_adv * suploss_adv \
                    + lambda_mixup * mixup_loss \
                    + lambda_in * (in_loss + rloss) \
                    + lambda_con * (conloss_lb + conloss_ulb) \
                    + lambda_cons * (consloss_ulb + consloss_ulb_aug) \
                    + lambda_proto * (suploss_proto + suploss_proto_ulb) \
                    + lambda_metric * (metric_loss_lb + metric_loss_ulb) \
                    + lambda_kd * (suploss_kd + lambda_ukd * suploss_kd_ulb + suploss_feats_kd)
            else:
                loss = suploss_lb \
                    + lambda_kd * (suploss_kd + suploss_feats_kd) \
                    + lambda_con * conloss_lb  
                                 
                
            loss.backward()
            tg_optimizer.step()
            tg_lr_scheduler.step()
            
            train_loss += loss.item()
            train_suploss_lb += suploss_lb.item()
            train_conloss_lb += conloss_lb.item() if use_conloss else 0.0
            train_metric_loss_lb += metric_loss_lb.item() if use_conloss and use_metric_loss else 0.0
            train_suploss_kd += suploss_kd.item() if iteration > start_iteration and old_mask.sum() > 0 else 0.0
            train_suploss_feats_kd += suploss_feats_kd.item() if iteration > start_iteration and use_feats_kd and use_lb_kd and old_mask.sum() > 0 else 0.0
            train_suploss_adv += suploss_adv.item() if iteration > start_iteration and epoch >= warmup_epochs and not skip and use_adv and mask.sum().item() >= 10*old_cn else 0.0
            train_consloss_ulb += consloss_ulb.item() if include_unlabel and epoch >= warmup_epochs and not skip else 0.0
            train_consloss_ulb_aug += consloss_ulb_aug.item() if include_unlabel and epoch != 0 and epoch >= warmup_epochs and use_ulb_aug and iteration > start_iteration and not skip else 0.0
            train_conloss_ulb += conloss_ulb.item() if include_unlabel and epoch >= warmup_epochs and not no_use_conloss_on_ulb and not skip else 0.0
            train_metric_loss_ulb += metric_loss_ulb.item() if include_unlabel and epoch >= warmup_epochs and use_metric_loss and not no_use_conloss_on_ulb and not skip else 0.0
            train_suploss_kd_ulb += suploss_kd_ulb.item() if include_unlabel and epoch >= warmup_epochs and iteration > start_iteration and use_ulb_kd and not skip else 0.0
            train_mixup_loss += mixup_loss.item() if include_unlabel and epoch >= warmup_epochs and iteration > start_iteration and use_mix_up and not mask.bool().all() else 0.0
            train_inloss_ulb += in_loss.item() if include_unlabel and epoch != 0 and epoch >= warmup_epochs and iteration > start_iteration and use_sim else 0.0
            train_rloss_ulb += rloss.item() if include_unlabel and epoch != 0 and epoch >= warmup_epochs and iteration > start_iteration and use_sim and me_max else 0.0
            train_suploss_proto += suploss_proto.item() if proto_clissifier else 0.0
            train_suploss_proto_ulb += suploss_proto_ulb.item() if proto_clissifier and include_unlabel and epoch >= warmup_epochs and not skip else 0.0
            train_util_ratio += mask.mean().item() if include_unlabel and epoch >= warmup_epochs else 0.0  
            train_n_util_ratio += n_mask.mean().item() if include_unlabel and epoch >= warmup_epochs else 0.0
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            writer.add_scalar('Train_Stage/SupLoss_LB', suploss_lb.item(), epoch * len(trainloader) + batch_idx)
            writer.add_scalar("Train_Stage/LR", tg_lr_scheduler.get_last_lr()[0], epoch)
        
            if use_conloss:
                writer.add_scalar('Train_Stage/ConLoss_LB', conloss_lb.item(), epoch * len(trainloader) + batch_idx)
                if use_metric_loss:
                    writer.add_scalar('Train_Stage/MetricLoss_LB', metric_loss_lb.item(), epoch * len(trainloader) + batch_idx)
            
            if proto_clissifier:
                writer.add_scalar('Train_Stage/SupLoss_Proto', suploss_proto.item(), epoch * len(trainloader) + batch_idx)
                if include_unlabel and epoch >= warmup_epochs:
                    writer.add_scalar('Train_Stage/SupLoss_Proto_ULB', suploss_proto_ulb.item(), epoch * len(trainloader) + batch_idx)

            if iteration > start_iteration:
                writer.add_scalar('Train_Stage/SupLoss_KD', suploss_kd.item() if old_mask.sum() else 0.0, epoch * len(trainloader) + batch_idx)
                
                if use_feats_kd and use_lb_kd:
                    writer.add_scalar('Train_Stage/SupLoss_Feats_KD', suploss_feats_kd.item() if old_mask.sum() else 0.0, epoch * len(trainloader) + batch_idx)

            if include_unlabel and epoch >= warmup_epochs and not skip:
                writer.add_scalar('Train_Stage/ConsLoss_ULB', consloss_ulb.item(), epoch * len(trainloader) + batch_idx)
                writer.add_scalars('Train_Stage/Ratio_ULB', {'p_mask': mask.mean().item(), 'n_mask': n_mask.mean().item()}, epoch * len(trainloader) + batch_idx)
                writer.add_scalars('Train_Stage/Pseusdo_Acc', {'mask_acc': mask_pseudo_acc if mask.bool().any() else 0.0, 'no_mask_acc': no_mask_pseudo_acc if not mask.bool().all() else 0.0, 'acc': pseudo_acc}, epoch * len(trainloader) + batch_idx)
                
                if use_sim and epoch != 0:
                    writer.add_scalar('Train_Stage/InLoss_ULB', in_loss.item(), epoch * len(trainloader) + batch_idx)
                    if me_max:
                        writer.add_scalar('Train_Stage/Rloss', rloss.item(), epoch * len(trainloader) + batch_idx)

                if iteration > start_iteration and use_ulb_kd:
                    writer.add_scalar('Train_Stage/SupLoss_KD_ULB', suploss_kd_ulb.item(), epoch * len(trainloader) + batch_idx)
                
                if iteration > start_iteration and use_mix_up and not mask.bool().all():
                    writer.add_scalar('Train_Stage/Mixup_Loss', mixup_loss.item(), epoch * len(trainloader) + batch_idx)

                if iteration > start_iteration and use_adv and mask.sum().item() > adv_num:
                    writer.add_scalar('Train_Stage/SupLoss_ADV', suploss_adv.item(), epoch * len(trainloader) + batch_idx)
                    writer.add_scalar('Train_Stage/Adv_Num', adv_data_num, epoch * len(trainloader) + batch_idx)
                    
                if not no_use_conloss_on_ulb:
                    writer.add_scalar('Train_Stage/ConLoss_ULB', conloss_ulb.item(), epoch * len(trainloader) + batch_idx)
                    if use_metric_loss:
                        writer.add_scalar('Train_Stage/MetricLoss_ULB', metric_loss_ulb.item(), epoch * len(trainloader) + batch_idx)
                
                if iteration > start_iteration and use_ulb_aug and epoch != 0:
                    writer.add_scalar('Train_Stage/ConsLoss_ULB_Aug', consloss_ulb_aug.item() if epoch != 0 else 0.0, epoch * len(trainloader) + batch_idx) 
                    
                    if not mask.bool().all():
                        writer.add_scalars('Train_Stage/Ulb_Class_Num', {'old': old_class_num, 'new': new_class_num}, epoch * len(trainloader) + batch_idx) 
                        writer.add_scalars('Train_Stage/Acc_ULB_Aug', {'proto':  ulb_aug_new_acc, 'classifier': ulb_new_acc}, epoch * len(trainloader) + batch_idx) 

        if update_proto:
            prototypes_old, prototypes_new, pro = get_proto(trainloader, tg_model, old_cn, device, False)
        
        if include_unlabel and epoch >= warmup_epochs and not use_sim and not use_proto:
            mean_pseudo_label = torch.stack(mean_pseudo_label).mean(0)
            writer.add_text('Train Stage/Pseudo distribution', str(mean_pseudo_label.cpu().numpy()), epoch)

        test_loss, test_acc, test_loss_session, test_acc_session, test_old_acc, test_new_acc = validate(tg_model, testloader, device, weight_per_class, old_cn, nb_cl_fg, nb_cl)
        
        writer.add_scalars("Training Stage Accuracy", {"Train": 100.*correct/total, "Test": test_acc}, epoch)
        writer.add_scalars("Training Stage Loss", {"Train": train_loss / (batch_idx + 1), "Test": test_loss}, epoch)
        writer.add_scalars("Test Stage Accuracy", {"Old": test_old_acc, "New": test_new_acc}, epoch)
                                                                                                                                                                                                    
        if include_unlabel:
            pseudo_label_acc_p, pseudo_labels_ratio, pseudo_label_acc_s = test_pseudo_acc(tg_model, ssl_trainloader, device, old_cn, total_cn, 
                                                                                        prototypes_new, p_cutoff=0.0, unlabels_predict_mode=unlabels_predict_mode)
            writer.add_scalars("Training Stage Pseudo Accuracy", {"Acc_p": pseudo_label_acc_p, "Acc_s": pseudo_label_acc_s}, epoch)
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-3])
        cumul_acc = compute_accuracy_train(tg_model, tg_feature_model, pro, testloader, device=prototypes_ref_old.device)
        if cumul_acc > best_acc:
            print('Epoch: {}, Best: {}'.format(epoch, cumul_acc)) 
            best_acc = cumul_acc
            torch.save(tg_model, './checkpoint/{}_best_model_session_{}.pth'.format(ckp_prefix, iteration))
        if epoch % 10 == 0 or epoch == epochs-1:
            
            if include_unlabel and epoch >= warmup_epochs:
                if not use_sim and not use_proto:
                    print('Epoch: {}, Pseudo distribution: {}'.format(epoch, mean_pseudo_label.cpu().numpy()))
                print('Epoch: {}, use_ratio: {}, acc_classifier: {}, acc_proto: {}'.format(epoch, pseudo_labels_ratio, pseudo_label_acc_p, pseudo_label_acc_s))
            
            print('Epoch: {}, Trainset: {}, Unlabel_Trainset: {}, Lr: {}, P_cutoff: {}, Q_cutoff: {}, Lambda_KD: {}, Lambda_CON: {}, Lambda_CONS: {}, Lambda_ULB_KD: {}, Lambda_CAT: {}, Lambda_UKD: {}'.format(epoch, len(trainset), \
                    len(ssl_trainloader.dataset) if include_unlabel else 0, tg_lr_scheduler.get_last_lr()[0], p_cutoff, q_cutoff, lambda_kd, lambda_con, lambda_cons, (old_cn//(total_cn-old_cn)), lambda_cat, lambda_ukd))
            print('Epoch: {}, SupLoss_LB: {:.4f}, SupLoss_KD: {:.4f}, SupLoss_Feats_KD: {:.4f}, SupLoss_ADV: {:.4f}, MixupLoss: {:.4f}, ConLoss_LB: {:.4f}, Metric_loss_LB: {:.4f}, ConsLoss_ULB: {:.4f}, ConsLoss_ULB_Aug: {:.4f}, SupLoss_KD_ULB: {:.4f}, ConLoss_ULB: {:.4f}, MetricLoss_ULB: {:.4f}, SupLoss_Proto_LB: {:.4f}, SupLoss_Proto_ULB: {:.4f}, InLoss_ULB: {:.4f}, RLoss: {:.4f}, Loss: {:.4f} Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Test Seesion Loss: {:.4f}, Test Session Acc: {:.4f}'.format(epoch, \
                    train_suploss_lb / (batch_idx+1), train_suploss_kd / (batch_idx+1),  train_suploss_feats_kd / (batch_idx+1), train_suploss_adv  / (batch_idx+1), train_mixup_loss / (batch_idx+1), train_conloss_lb / (batch_idx+1), train_metric_loss_lb / (batch_idx+1), train_consloss_ulb / (batch_idx+1), train_consloss_ulb_aug / (batch_idx+1),
                    train_suploss_kd_ulb / (batch_idx+1), train_conloss_ulb / (batch_idx+1), train_metric_loss_ulb / (batch_idx+1), train_suploss_proto / (batch_idx+1), train_suploss_proto_ulb / (batch_idx+1), train_inloss_ulb / (batch_idx+1), train_rloss_ulb  / (batch_idx+1), train_loss / (batch_idx+1), 100. * correct / total, test_loss, test_acc, test_loss_session, test_acc_session))
    
    
    loss, acc, loss_session, acc_session, old_acc, new_acc = validate(tg_model, testloader, device, weight_per_class, old_cn, nb_cl_fg, nb_cl)
    print('Test set: {} Test Loss: {:.4f} Acc: {:.4f} Test Session Loss: {:.4f} Session Acc: {:.4f}'.format(len(testloader), loss, acc, loss_session, acc_session))
    writer.close()
    return tg_model

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        return F.cross_entropy(logits, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape, print(logits.shape, targets.shape)
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return nll_loss


def consistency_loss(logits_w, logits_s, feats_ulb, text_anchor, old_cn, total_cn, distri, 
                     gt, prototypes_new, name='ce', T=0.5, p_cutoff=0.0, use_hard_labels=True,
                     use_proto=False, use_da=False, no_use_conloss=False, unlabels_predict_mode='cosine'):
    assert name in ['ce', 'L2']

    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        
        if use_proto:
            if unlabels_predict_mode == 'cosine':
                cosine_scores = F.cosine_similarity(prototypes_new.unsqueeze(0).repeat(len(feats_ulb), 1, 1),
                                                    feats_ulb.unsqueeze(1).repeat(1, len(prototypes_new), 1), 2) / 0.1
                pseudo_label = torch.softmax(cosine_scores, dim=1)
                max_probs, max_idx = torch.max(pseudo_label, dim=1)
                mask = max_probs.ge(p_cutoff).float()
                predicted_classes = torch.argmax(cosine_scores, dim=1)  # (batch_size,)
            elif unlabels_predict_mode == 'sqeuclidean':
                class_means_squared = torch.sum(prototypes_new**2, dim=1, keepdim=True)  # (num_classes, 1)
                outputs_feature_squared = torch.sum(feats_ulb**2, dim=1, keepdim=True).T  # (1, batch_size)
                dot_product = torch.matmul(prototypes_new, feats_ulb.T)  # (num_classes, batch_size)
                squared_distances = class_means_squared + outputs_feature_squared - 2 * dot_product  # (num_classes, batch_size)
                pseudo_label = torch.softmax(-torch.sqrt(squared_distances.T), dim=1)  # (num_classes, batch_size)
                max_probs, max_idx = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(p_cutoff).float()
                predicted_classes = torch.argmin(squared_distances, dim=0)  # (batch_size,)
            else:
                raise ValueError('unlabels_predict_mode: {} not supported'.format(unlabels_predict_mode))
        else:
            pseudo_label = torch.softmax(logits_w[:, old_cn:total_cn], dim=-1)
            
            if use_da:
                pseudo_label = pseudo_label / distri
                pseudo_label = pseudo_label / pseudo_label.sum(dim=1, keepdim=True)

            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff).float()
            indices = mask.nonzero(as_tuple=True)[0]
            predicted_classes = max_idx
        
        predicted_classes = predicted_classes + old_cn
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, predicted_classes, use_hard_labels, reduction='none') 
        else:
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask

        if not no_use_conloss:
            feats_ulb_masked = feats_ulb
            scores = F.cosine_similarity(text_anchor.unsqueeze(0).repeat(len(feats_ulb_masked), 1, 1),
            feats_ulb_masked.unsqueeze(1).repeat(1, len(text_anchor), 1), 2) / 0.1
            conloss_ulb = F.cross_entropy(scores, predicted_classes.long())
        else:
            conloss_ulb = 0.0

        return masked_loss.mean(), conloss_ulb

    else:
        assert Exception('Not Implemented consistency_loss')


def ce_loss_raw(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


def consistency_loss_raw(logits, targets, name='ce', mask=None):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss_raw(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()


def get_proto(trainloader, tg_model, old_cn, device, normalize=True):
    tg_model.eval()
    class_features = {}
    class_counts = {}

    for batch_idx, (indexs, inputs, inputs_s, targets, flags, on_flags) in enumerate(trainloader):
        # 将输入和目标移动到设备上
        inputs, inputs_s, targets, flags, on_flags = inputs.to(device), inputs_s.to(device), targets.to(device), flags.to(device), on_flags.to(device)  
        if len(inputs) == 1:
            continue
        # 获取模型输出的特征
        with torch.no_grad():
            # 原型改动
            outputs, raw_feats, feats, session_outputs = tg_model(inputs, return_feats=True)
            # outputs, feats, raw_feats, session_outputs = tg_model(inputs, return_feats=True)
        
        # 遍历当前 batch 的所有样本
        for i in range(len(targets)):
            label = targets[i].item()  # 获取当前样本的标签
            feature = feats[i]  # 获取当前样本的特征向量
            
            # 如果这个类还没有记录过特征，初始化累加器
            if label not in class_features:
                class_features[label] = torch.zeros_like(feature)
                class_counts[label] = 0
            
            # 累加特征向量
            class_features[label] += feature
            class_counts[label] += 1

    # 计算每个类的特征均值并保存为 tensor
    prototypes = []
    prototypes_new = []
    prototypes_old = []
    for label in sorted(class_features.keys()):
        # 只保留类别索引大于等于 old_cn 的类
        class_mean = class_features[label] / class_counts[label]    
        if normalize:
            class_mean = F.normalize(class_mean, p=2, dim=0)
        prototypes.append(class_mean)
        if label >= old_cn:
            prototypes_new.append(class_mean)
        else:
            prototypes_old.append(class_mean)
    
    if len(prototypes_old) == 0:
        prototypes_old = torch.tensor([])
    else:
        prototypes_old = torch.stack(prototypes_old, dim=0)
    
    if len(prototypes_new) == 0:
        prototypes_new = torch.tensor([])
    else:
        prototypes_new = torch.stack(prototypes_new, dim=0)
    
    prototypes = torch.stack(prototypes, dim=0)
    
    prototypes_old, prototypes_new, prototypes = prototypes_old.to(device), prototypes_new.to(device), prototypes.to(device)

    return prototypes_old, prototypes_new, prototypes


import numpy as np
import torch

def fill_pro_list(pro_list, tg_model, val_loader, device, k, old_cn):
    tg_model.eval()
    # 存储所有输入特征和标签
    all_feats = []
    all_index = []
    all_gt = []
    all_inputs = []
    all_outputs = []
    dataset = val_loader.dataset
    # 计算val_loader中所有输入的特征
    with torch.no_grad():
        for batch_idx, (index, inputs, _, gt, _, _) in enumerate(val_loader):
            inputs = inputs.to(device)
            gt = gt.to(device)
            outputs, _, feats, _ = tg_model(inputs, return_feats=True)
            outputs = torch.softmax(outputs, dim=1)  # 转为概率

            # 存储特征、标签和输入数据
            # all_feats.extend(feats.cpu().numpy())
            all_gt.extend(gt.cpu().numpy())
            all_inputs.extend(inputs.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            all_index.extend(index.cpu().numpy())
    
    # 转换为numpy array，便于处理
    # all_feats = np.array(all_feats)
    all_gt = np.array(all_gt)
    all_inputs = np.array(all_inputs)
    all_outputs = np.array(all_outputs)
    all_index = np.array(all_index)

    # 对每个类进行处理，选择置信度最高的 K 个样本
    for label in range(old_cn, all_outputs.shape[1]):  # 只处理 old_cn 之后的类别
        # 获取该类别的所有样本的置信度
        class_confidences = all_outputs[:, label]
        
        # 选择置信度最高的 K 个样本
        top_k_indices = np.argsort(class_confidences)[-k:]

        correct_count = 0  # 统计正确分类的样本数
        for idx in top_k_indices:
            # 检查预测的最高置信度的类别是否匹配真实标签
            if all_gt[idx] == label:
                correct_count += 1
            
            # 将输入数据格式转换并添加到 pro_list 中，确保不重复
            # new_proto = all_inputs[idx]
            # new_proto = (new_proto * 255).clip(0, 255).astype(np.uint8)
            # if new_proto.tolist() not in [p.tolist() for p in pro_list[label]]:
            #     pro_list[label] = np.concatenate((pro_list[label], np.transpose(new_proto, (1, 2, 0))[np.newaxis, :]), axis=0)
        selected_index = all_index[top_k_indices]
        pro_list[label] = np.concatenate((pro_list[label], dataset.data[selected_index]), axis=0)

        # 计算并打印当前类别的准确率
        accuracy = correct_count / k
        print(f"Accuracy for class {label} neighbors: {accuracy:.2%}")

    return pro_list

def test_pseudo_acc(tg_model, val_loader, device, old_cn, total_cn, prototypes_new, 
                    p_cutoff=0.0, unlabels_predict_mode='cosine'):
    acc_p = 0
    acc_s = 0
    totalnum_p = 0
    totalnum = 0
    tg_model.eval()
    for batch_idx, (inputs, _, gt) in enumerate(val_loader):
        inputs = inputs.to(device)
        gt = gt.to(device)
        outputs, raw_feats, feats, session_outputs = tg_model(inputs, return_feats=True)
        out_prob = F.softmax(outputs, dim=1)
        # [[session1],[session2],[session3],.....]
        outputs_new = F.softmax(outputs[:, old_cn: total_cn], dim=1)

        max_value, max_idx = torch.max(outputs_new, dim=1)
        max_value_all, max_idx_all = torch.max(out_prob, dim=1)
        mask = max_value.ge(p_cutoff)    
        mask = mask.float()            
        maskindex_total = torch.where(mask==1)[0]
        totalnum += mask.numel()
        totalnum_p += len(maskindex_total)
        if not len(maskindex_total)==0:
            acc_p += (max_idx[maskindex_total]+old_cn).eq(gt[maskindex_total]).sum().cpu().item()  
        
        feats = F.normalize(feats, p=2, dim=1)

        if unlabels_predict_mode == 'sqeuclidean':
            class_means_squared = torch.sum(prototypes_new**2, dim=1, keepdim=True)  # (num_classes, 1)
            outputs_feature_squared = torch.sum(feats**2, dim=1, keepdim=True).T  # (1, batch_size)
            dot_product = torch.matmul(prototypes_new, feats.T)  # (num_classes, batch_size)
            squared_distances = class_means_squared + outputs_feature_squared - 2 * dot_product  # (num_classes, batch_size)
            outputs_new_proto = torch.softmax(-torch.sqrt(squared_distances.T), dim=1)  # (num_classes, batch_size)
            max_value_proto, max_idx_proto = torch.max(outputs_new_proto, dim=1)
            predicted_classes = torch.argmin(squared_distances, dim=0)  # (batch_size,)
        elif unlabels_predict_mode == 'cosine':
            cosine_scores = F.cosine_similarity(prototypes_new.unsqueeze(0).repeat(len(feats), 1, 1),
                                        feats.unsqueeze(1).repeat(1, len(prototypes_new), 1), 2) / 0.1
            outputs_new_proto = torch.softmax(cosine_scores, dim=1)  # (num_classes, batch_size)
            max_value_proto, max_idx_proto = torch.max(outputs_new_proto, dim=1)
            predicted_classes = torch.argmax(cosine_scores, dim=1)  # (batch_size,)
        else:
            raise ValueError('unlabels_predict_mode: {} not supported'.format(unlabels_predict_mode))
        assert max_idx_proto.eq(predicted_classes).sum().cpu().item() == len(feats)
        predicted_classes = predicted_classes + old_cn
        acc_s += predicted_classes.eq(gt).sum().cpu().item()  

    if totalnum_p==0:
        pseudo_label_acc = 0
    else:
        pseudo_label_acc = acc_p/totalnum_p

    return 100*pseudo_label_acc, totalnum_p/totalnum, 100*acc_s/totalnum


def validate(tg_model, testloader, device, weight_per_class, old_cn, nb_cl_fg=None, nb_cl=None):
    # eval
    tg_model.eval()
    test_loss = 0
    test_loss_session = 0
    correct = 0
    correct_session = 0
    total = 0

    predicted_list = []
    gt_list = []
    
    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _, session_outputs = tg_model(inputs, return_feats=True)
            loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            predicted_list.append(predicted.cpu().numpy())
            gt_list.append(targets.cpu().numpy())

            # if session_outputs is not None:
            #     session_targets = get_session_labels(targets, nb_cl_fg, nb_cl)
            #     loss_session = nn.CrossEntropyLoss()(session_outputs, session_targets)
            #     test_loss_session += loss_session.item()
            #     _, predicted_session = session_outputs.max(1)
            #     correct_session += predicted_session.eq(session_targets).sum().item()
    
    predicted_list = np.concatenate(predicted_list)
    gt_list = np.concatenate(gt_list)

    old_mask = gt_list < old_cn
    new_mask = gt_list >= old_cn
    if old_mask.sum() > 0:
        old_acc = (predicted_list[old_mask] == gt_list[old_mask]).mean()
    else:
        old_acc = 0.0
    if new_mask.sum() > 0:
        new_acc = (predicted_list[new_mask] == gt_list[new_mask]).mean()
    else:
        new_acc = 0.0

    return test_loss/(batch_idx+1), 100.*correct/total, test_loss_session/(batch_idx+1), 100.*correct_session/total, 100.*old_acc, 100.*new_acc
     
    
def get_session_labels(class_labels: torch.tensor, nb_cl_fg: int, nb_cl: int):
    session_labels = torch.ones_like(class_labels) * -1
    for i in range(len(class_labels)):
        if class_labels[i] < nb_cl_fg:
            session_labels[i] = 0
        else:
            session_labels[i] = (class_labels[i] - nb_cl_fg) // (nb_cl) + 1
    return session_labels.to(class_labels)


def gen_adv_data(ref_model, adv_x, ref_feats, gt, old_cn, prototypes_ref_old, device, adv_num, adv_alpha, adv_epochs, x_min, x_max):

    gen_x = []
    gen_y = []
    assert len(adv_x) == len(ref_feats)
    adv_num = len(adv_x) // old_cn // 2

    for c in range(old_cn):
        d = torch.cdist(ref_feats, prototypes_ref_old[c].unsqueeze(0)).squeeze()
        closest = torch.argsort(d)[:adv_num].cpu()
        x_top = adv_x[[closest]]
        y_top = gt[[closest]]
        

        idx_dataset = torch.utils.data.TensorDataset(x_top, torch.ones(x_top.size(0), dtype=torch.long) * c)
        loader = torch.utils.data.DataLoader(idx_dataset, batch_size=int(adv_num), shuffle=False)

        attack = Attack(ref_model, adv_alpha, loader, prototypes_ref_old, device, adv_epochs, x_min, x_max, c)
        
        x_, y_ = attack.run()
        
        gen_x.append(x_.detach())       
        gen_y.append(y_.detach())
    
    gen_x = torch.cat(gen_x, dim=0)
    gen_y = torch.cat(gen_y, dim=0)

    return gen_x, gen_y


def save_tensor_to_img(x_top, tg, path):
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    # 计算反归一化参数
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]

    # 定义反归一化变换
    inv_normalize = transforms.Normalize(mean=inv_mean, std=inv_std)

    # 定义转换为 PIL 图像的变换
    to_pil = transforms.ToPILImage()

    # 假设 x_top 是加载到 GPU 上的图片 Tensor
    # 将 Tensor 从 GPU 移动到 CPU
    x_top_cpu = x_top.cpu()

    # 处理每一张图片
    for i in range(x_top_cpu.size(0)):
        img_tensor = x_top_cpu[i]
        
        # 反归一化
        img_tensor = inv_normalize(img_tensor)
        
        # 转换为 PIL 图像
        img = to_pil(img_tensor)
        
        # 保存图像
        img.save(os.path.join(path, f'image_{i}_tg_{tg[i].item()}.png'))
    