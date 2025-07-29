#!/usr/bin/env python
# coding=utf-8
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from utils_pytorch import *
from dataloder import BaseDataset, UnlabelDataset, ReservedUnlabelDataset
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pdb
import math
from torch.utils.data import BatchSampler, RandomSampler
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter

def incremental_train_and_eval(args, base_lamda, adapt_lamda, u_t, label2id, uncertainty_distillation, 
                               prototypes, prototypes_flag, prototypes_on_flag, update_unlabeled, 
                               epochs, method, unlabeled_num, unlabeled_iteration, unlabeled_num_selected, 
                               train_batch_size, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, 
                               testloader, iteration, start_iteration, T, beta, unlabeled_data, unlabeled_gt, nb_cl_fg, 
                               nb_cl, trainset, image_size, text_anchor, use_conloss=True, include_unlabel=True,
                               con_margin=0.2, hard_negative=False, fix_bn=False, weight_per_class=None, 
                               device=None, use_da=False, use_proto=False, update_proto=False, u_ratio=1,lambda_kd=1.0, 
                               lambda_con=1.0, lambda_cons=1.0, lambda_in=1.0, lambda_reg=1.0, lambda_session=1.0, use_proto_classifier=False, 
                               kd_only_old=False, u_iter=100, no_use_conloss_on_ulb=False, unlabels_predict_mode='sqeuclidean', use_hard_labels=True,
                               use_sim=False, smoothing_alpha=0.7, p_cutoff=0.0, use_ulb_kd=False, use_srd=False, use_session_labels=False, use_ulb_aug=False):
    N = 128
    writer = SummaryWriter(log_dir='checkpoint/logs/{}/{}'.format(args.ckp_prefix, iteration))
    
    if use_conloss:
        text_anchor = text_anchor.to(device)

    if iteration > start_iteration:
        unlabeled_trainset = UnlabelDataset(image_size, dataset=args.dataset, autoaug=args.autoaug)
        unlabeled_trainset.data = unlabeled_data
        unlabeled_trainset.targets = unlabeled_gt
        unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=train_batch_size,
                                                            shuffle=True, num_workers=4) 
        ssl_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=u_ratio*train_batch_size, 
                                                      shuffle=True, num_workers=4)
        # print("unlabeled dataset trans: {}, \nstrong_trans: {}".format(unlabeled_trainset.transform,
        #                                                                unlabeled_trainset.strong_transform))
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features

        if use_sim:
            ema_bank = 0.7
            smoothing_alpha=0.7
            use_ema_teacher = False
            mem_bank = torch.randn(512, len(trainset)).to(device)
            mem_bank = F.normalize(mem_bank, dim=0)
            labels_bank = torch.zeros(len(trainset), dtype=torch.long).to(device)
            mem_bank, labels_bank = mem_bank.detach(), labels_bank.detach()
            
            def update_bank(k, labels, index):
                if use_ema_teacher:
                    mem_bank[:, index] = k.t().detach()
                else:
                    mem_bank[:, index] = F.normalize(ema_bank * mem_bank[:, index] + (1 - ema_bank) * k.t().detach())
                labels_bank[index] = labels.detach()
        

    # train the model with labeled data
    for epoch in range(epochs):
        # train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        total = 0
        correct = 0
        train_loss = 0
        train_suploss_kd = 0
        train_suploss_lb = 0
        train_conloss_lb = 0
        train_suploss_reg = 0
        train_suploss_session = 0
        
        if epoch % 40 == 0:
            print('\nEpoch: %d, LR: ' % epoch, end='')
            print(tg_lr_scheduler.get_last_lr())
        
        for batch_idx, (indexs, inputs, inputs_s, targets, flags, on_flags) in enumerate(trainloader):
            tg_optimizer.zero_grad()
            inputs, inputs_s, targets, flags, on_flags = inputs.to(device), inputs_s.to(device), targets.to(device), flags.to(device), on_flags.to(device)
            assert len(inputs) == len(inputs_s)
            
            if len(inputs) == 1:
                continue

            outputs, raw_feats, feats, session_outputs = tg_model(inputs, return_feats=True)
            feats = F.normalize(feats, p=2, dim=1)
            
            if iteration == start_iteration:
                suploss_lb = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())
                # 将提取的视觉特征与text特征空间对齐
                if use_conloss:    
                    scores = F.cosine_similarity(text_anchor.unsqueeze(0).repeat(len(feats), 1, 1),
                                            feats.unsqueeze(1).repeat(1, len(text_anchor), 1), 2) / 0.1
                    conloss_lb = F.cross_entropy(scores, targets.long())
                else:
                    conloss_lb = 0.0
                loss = suploss_lb + lambda_con * conloss_lb
            else:
                ref_outputs, ref_raw_feats, _, _= ref_model(inputs, return_feats=True)
                suploss_lb = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())
                if use_session_labels:
                    session_targets = get_session_labels(targets, nb_cl_fg, nb_cl)
                    suploss_lb_session = nn.CrossEntropyLoss()(session_outputs, session_targets.long())
                else:
                    suploss_lb_session = 0.0 

                if uncertainty_distillation:
                    ##################################
                    # uncertainty-aware distillation #
                    ##################################
                    out_prob = []
                    for _ in range(10):
                        #Gaussian noise
                        noise = torch.clamp(torch.randn_like(inputs) * 0.01, -0.02, 0.02)
                        inputs_noise = inputs + noise.to(device)
                        outputs_noise = ref_model(inputs_noise)
                        out_prob.append(F.softmax(outputs_noise, dim=1))
                    out_prob = torch.stack(out_prob)
                    out_std = torch.std(out_prob, dim=0)
                    out_prob = torch.mean(out_prob, dim=0)
                    max_value, max_idx = torch.max(out_prob, dim=1)
                    max_std = out_std.gather(1, max_idx.view(-1, 1))
                    max_std_sorted, std_indices = torch.sort(max_std, descending=False)
                    max_std = max_std.squeeze(1).detach().cpu().numpy()

                    outputs_cp = outputs
                    outputs = outputs.detach().cpu().numpy()
                    ref_outputs = ref_outputs.detach().cpu().numpy()
                    
                    idx_del = []
                    for idx in range(len(max_std)):
                        if max_std[idx] > max_std_sorted[int(u_t * len(max_std))]:
                            if flags[idx] == 0:
                                idx_del.append(idx)
                    
                    outputs = np.delete(outputs, idx_del, axis = 0)
                    outputs = torch.from_numpy(outputs)

                    ref_outputs = np.delete(ref_outputs, idx_del, axis = 0)
                    ref_outputs = torch.from_numpy(ref_outputs)
                    if adapt_lamda:
                        cur_lamda = base_lamda * 1 / u_t *  math.sqrt(num_old_classes / nb_cl)
                    else:
                        cur_lamda = base_lamda
                    
                    suploss_kd = cur_lamda * nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
                                       F.softmax(ref_outputs.detach() / T, dim=1)) * T * T * beta * num_old_classes
                    suploss_reg = 0.0
                else:
                    if use_srd:
                        old_mask = targets < num_old_classes
                        tg_outputs = ref_model.get_logits(raw_feats)
                        suploss_kd = nn.MSELoss()(tg_outputs[old_mask], ref_outputs.detach()[old_mask]) * 10
                        suploss_reg = nn.L1Loss()(raw_feats[old_mask], ref_raw_feats.detach()[old_mask])
                    else:
                        if kd_only_old:
                            old_mask = targets < num_old_classes
                            suploss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[old_mask][:, :num_old_classes] / T, dim=1),
                                            F.softmax(ref_outputs[old_mask].detach() / T, dim=1)) * T * T * beta * num_old_classes
                        else:
                            suploss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
                                            F.softmax(ref_outputs.detach() / T, dim=1)) * T * T * beta * num_old_classes
                        suploss_reg = 0.0 
                # 将提取的视觉特征与text特征空间对齐
                if use_conloss:
                    scores = F.cosine_similarity(text_anchor.unsqueeze(0).repeat(len(feats), 1, 1),
                                            feats.unsqueeze(1).repeat(1, len(text_anchor), 1), 2) / 0.1
                    conloss_lb = F.cross_entropy(scores, targets.long()) 
                else:
                    conloss_lb = 0.0
                
                loss = suploss_lb + lambda_kd * suploss_kd + lambda_con * conloss_lb + lambda_reg * suploss_reg + lambda_session * suploss_lb_session
                
            loss.backward()
            tg_optimizer.step()
            # tg_lr_scheduler.step()
            
            train_loss += loss.item()
            train_suploss_lb += suploss_lb.item()
            train_conloss_lb += conloss_lb.item() if use_conloss else 0.0
            train_suploss_kd += suploss_kd.item() if iteration > start_iteration else 0.0
            train_suploss_reg += suploss_reg.item() if iteration > start_iteration and use_srd else 0.0
            train_suploss_session += suploss_lb_session.item() if iteration > start_iteration and use_session_labels else 0.0

            if uncertainty_distillation and iteration > start_iteration:
                _, predicted = outputs_cp.max(1)
            else:
                _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            writer.add_scalar('Train/SupLoss_LB', suploss_lb.item(), epoch * len(trainloader) + batch_idx)
        
            if use_conloss:
                writer.add_scalar('Train/ConLoss_LB', conloss_lb.item(), epoch * len(trainloader) + batch_idx)
            
            if iteration > start_iteration:
                writer.add_scalar('Train/SupLoss_KD', suploss_kd.item(), epoch * len(trainloader) + batch_idx)
            
                if use_srd:
                    writer.add_scalar('Train/SupLoss_REG', suploss_reg.item(), epoch * len(trainloader) + batch_idx)

                if use_session_labels:
                    writer.add_scalar('Train/SupLoss_SESSION', suploss_lb_session.item(), epoch * len(trainloader) + batch_idx)

        tg_lr_scheduler.step()
        test_loss, test_acc, test_loss_session, test_acc_session = validate(tg_model, testloader, device, weight_per_class, nb_cl_fg, nb_cl)
        
        writer.add_scalar("LR", tg_lr_scheduler.get_last_lr()[0], epoch)
        writer.add_scalars("Accuracy", {"Train": 100.*correct/total, "Test": test_acc}, epoch)
        writer.add_scalars("Loss", {"Train": train_loss / (batch_idx + 1), "Test": test_loss}, epoch)
        
        if use_session_labels:
            writer.add_scalar("Session Accuracy", test_acc_session, epoch)
            writer.add_scalar("Session Loss",  test_loss_session, epoch)

        if epoch % 40 == 0 or epoch == epochs-1:    
            if iteration == start_iteration:
                print('Trainset: {}, SupLoss_LB: {:.4f}, ConLoss_LB: {:.4f}, Loss: {:.4f},  Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(len(trainloader.dataset), \
                    train_suploss_lb / (batch_idx + 1), train_conloss_lb / (batch_idx + 1),
                    train_loss / (batch_idx + 1), 100. * correct / total, test_loss, test_acc))
            else:
                print('Trainset: {}, SupLoss_LB: {:.4f}, SupLoss_KD: {:.4f}, SupLoss_REG: {:.4f}, SupLoss_SESSION: {:.4f}, ConLoss_LB: {:.4f}, Loss: {:.4f} Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Test Session Loss: {:.4f}, Test Session Acc: {:.4f}'.format(len(trainloader.dataset), \
                        train_suploss_lb / (batch_idx + 1), train_suploss_kd / (batch_idx + 1), train_suploss_reg / (batch_idx + 1), train_suploss_session / (batch_idx + 1),
                        train_conloss_lb / (batch_idx + 1), train_loss / (batch_idx + 1), 100. * correct / total, test_loss, test_acc, test_loss_session, test_acc_session))


    if use_proto_classifier:
        prototypes_tensor, _, pro = get_proto(trainloader, tg_model, old_cn, device, normalize=False)
        assert tg_model.fc.weight.size() == prototypes_tensor.size()
        tg_model.fc.weight.data.copy_(prototypes_tensor.to(device))
    
    # if add unlabeled data, start unlabeled iteration.
    total_unlabeled_selected = 0  # total number of unlabeled data selected so far.
    
    old_cn = iteration * nb_cl
    total_cn = (iteration + 1) * nb_cl

    if iteration > start_iteration and unlabeled_data is not None:   
        selected_unlabeld_data = None
        selected_unlabeld_targets = None
        selected_unlabeld_predicts = None
           
        prototypes_old, prototypes_new, pro = get_proto(trainloader, tg_model, old_cn, device) 
        prototypes_ref_old, prototypes_ref_new,  prototypes_ref= get_proto(trainloader, ref_model, old_cn, device) 
        
        if method == "self_train":
            trainset_1 = BaseDataset("train", image_size, label2id, dataset=args.dataset, autoaug=args.autoaug)
            trainset_1.data = trainset.data
            trainset_1.targets = trainset.targets

            for u_i in range(unlabeled_iteration):
                
                if total_unlabeled_selected < unlabeled_num_selected:
                    num_unlabeled = 10  # number of unlabeled data selected from every epoch.
                    num_unlabeled = min(num_unlabeled, 
                                        unlabeled_data.shape[0],
                                        unlabeled_num_selected - total_unlabeled_selected)
                    
                    if num_unlabeled < nb_cl:
                        break

                    selected_idx = []
                    unlabeled_selected = []
                    unlabeled_selected_l = []
                    # total max_values and max_stds
                    max_values = []
                    max_indices = []
                    max_indices_all = []
                    max_stds = []
                    outputs_unlabeled = []
                    gt_unlabeled = []
                    
                    # for class-balance self-train
                    for batch_idx, (inputs, _, gt) in enumerate(unlabeled_trainloader):
                        inputs = inputs.to(device)
                        gt = gt.to(device)
                        
                        if len(inputs) == 1:
                            continue

                        outputs = tg_model(inputs)
                        out_prob = F.softmax(outputs, dim=1)
                        # [[session1],[session2],[session3],.....]
                        outputs_new = out_prob[:, old_cn: total_cn]
                        max_value, max_idx = torch.max(outputs_new, dim=1)
                        max_value_all, max_idx_all = torch.max(out_prob, dim=1)
                        
                        if batch_idx == 0:
                            max_values = max_value
                            max_indices = max_idx
                            max_indices_all = max_idx_all
                            outputs_unlabeled = outputs
                            gt_unlabeled = gt
                        else:
                            max_values = torch.cat((max_values, max_value), 0)
                            max_indices = torch.cat((max_indices, max_idx), 0)
                            max_indices_all = torch.cat((max_indices_all, max_idx_all), 0)
                            outputs_unlabeled = torch.cat((outputs_unlabeled, outputs), 0)
                            gt_unlabeled = torch.cat((gt_unlabeled, gt), 0)

                    print('for class-balance selection')
                    for c_i in range(nb_cl):
                        idx_cl = [i for (i, value) in enumerate(max_indices) if value == c_i]
                        max_values_cl = max_values[idx_cl]
                        if len(idx_cl) <= int(num_unlabeled/nb_cl):
                            if c_i == 0:
                                same_indices = idx_cl
                            else:
                                same_indices = np.concatenate((same_indices, idx_cl), axis=0)
                        else:
                            idx_cl = np.array(idx_cl)
                            max_values_cl_sorted_idx = np.argsort(-max_values_cl.detach().cpu().numpy())  # descending order
                            selected_cl_idx = idx_cl[max_values_cl_sorted_idx[:int(num_unlabeled/nb_cl)]]
                            if c_i == 0:
                                same_indices = selected_cl_idx
                            else:
                                same_indices = np.concatenate((same_indices, selected_cl_idx), axis=0)

                    same_indices = same_indices.astype(int)
                    unlabeled_selected = unlabeled_data[same_indices]
                    gt_unlabeled_selected = gt_unlabeled[same_indices]
                    unlabeled_selected_l = old_cn + max_indices[same_indices]
                    num_unlabeled = len(same_indices)
                    selected_idx = same_indices
                    
                    print("select pseudo-labeling data acc: ", unlabeled_selected_l.eq(gt_unlabeled_selected).sum().cpu().item() / len(gt_unlabeled_selected))
                    print("u_iter {} selected {} ".format(u_i, len(unlabeled_selected)))
                    
                    if num_unlabeled > 0:
                        total_unlabeled_selected += num_unlabeled
                        unlabeled_data = np.delete(unlabeled_data, selected_idx, axis=0)
                        unlabeled_gt = np.delete(unlabeled_gt, selected_idx, axis=0)
                        unlabeled_selected = np.array(unlabeled_selected)
                        unlabeled_selected_l = np.array(unlabeled_selected_l.cpu().numpy())
                        gt_unlabeled_selected = np.array(gt_unlabeled_selected.cpu().numpy())
                        print('the total number of unlabeled data selected is {}, have {} unlabels data'.format(total_unlabeled_selected, len(unlabeled_data)))
                        
                        # add unlabeled data to prototypes and prototypes_flag for computing class-means
                        if update_unlabeled:
                            for i in range(len(unlabeled_selected_l)):
                                if len(unlabeled_selected[i].shape) > 1:
                                    prototypes[unlabeled_selected_l[i]] = np.concatenate([prototypes[unlabeled_selected_l[i]], np.expand_dims(unlabeled_selected[i], axis=0)])
                                else:
                                    prototypes[unlabeled_selected_l[i]] = np.append(prototypes[unlabeled_selected_l[i]], unlabeled_selected[i])
                                prototypes_flag[unlabeled_selected_l[i]] = np.append(prototypes_flag[unlabeled_selected_l[i]], 0)
                                prototypes_on_flag[unlabeled_selected_l[i]] = np.append(prototypes_on_flag[unlabeled_selected_l[i]], 0)

                        # add unlabeled data to trainset
                        ################################
                        trainset_1.data = np.concatenate([trainset_1.data, unlabeled_selected])
                        trainset_1.targets = np.concatenate([trainset_1.targets, unlabeled_selected_l])
                        sampler_x = RandomSampler(trainset_1, replacement=True, num_samples = u_iter * train_batch_size)
                        batch_sampler_x = BatchSampler(sampler_x, train_batch_size, drop_last=True)  # yield a batch of samples one time
                        trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_sampler=batch_sampler_x, num_workers=4)
                        # if args.dataset == "cub":
                        #     # train iter = 100
                        #     sampler_x = RandomSampler(trainset_1, replacement=True, num_samples = u_iter * train_batch_size)
                        #     batch_sampler_x = BatchSampler(sampler_x, train_batch_size, drop_last=True)  # yield a batch of samples one time
                        #     trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_sampler=batch_sampler_x, num_workers=4)
                        # else:
                        #     # train iter = trainsize / batchsize
                        #     trainloader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=train_batch_size, shuffle=True, num_workers=4)
                        
                        if selected_unlabeld_data is None:
                            selected_unlabeld_data = unlabeled_selected
                            selected_unlabeld_targets = gt_unlabeled_selected
                            selected_unlabeld_predicts = unlabeled_selected_l
                        else:
                            selected_unlabeld_data = np.concatenate([selected_unlabeld_data, unlabeled_selected])
                            selected_unlabeld_targets = np.concatenate([selected_unlabeld_targets, gt_unlabeled_selected])
                            selected_unlabeld_predicts = np.concatenate([selected_unlabeld_predicts, unlabeled_selected_l])

                        for epoch in range(10):
                            total = 0
                            correct = 0
                            ulb_total = 0
                            ulb_correct = 0
                            ulb_mask_total = 0
                            ulb_mask_correct = 0
                            train_loss = 0
                            train_suploss_lb = 0
                            train_conloss_lb = 0
                            train_suploss_kd = 0
                            train_suploss_reg = 0
                            train_suploss_session = 0
                            train_suploss_ulb_session = 0
                            train_conloss_ulb = 0
                            train_consloss_ulb = 0
                            train_suploss_kd_ulb = 0
                            train_consloss_ulb_aug = 0
                            train_inloss_ulb = 0
                            train_util_ratio = 0
                            tg_model.train()
                            
                            if fix_bn:
                                for m in tg_model.modules():
                                    if isinstance(m, nn.BatchNorm2d):
                                        m.eval()
                            
                            distri = []
                            ssl_iterator = iter(ssl_trainloader)
                            
                            for batch_idx, (inputs, inputs_s, targets) in enumerate(trainloader_1):
                            
                                inputs, inputs_s, targets = inputs.to(device), inputs_s.to(device), targets.to(device)
                                if len(inputs) == 1:
                                    continue
                                
                                tg_optimizer.zero_grad()
                                outputs, raw_feats, feats, session_outputs = tg_model(inputs, return_feats=True)
                                ref_outputs, ref_raw_feats, ref_feats, _ = ref_model(inputs, return_feats=True)
                                feats = F.normalize(feats, p=2, dim=1)
                                ref_feats = F.normalize(ref_feats, p=2, dim=1)

                                suploss_lb = nn.CrossEntropyLoss(weight_per_class)(outputs, targets.long())

                                if use_session_labels:
                                    session_targets = get_session_labels(targets, nb_cl_fg, nb_cl)
                                    suploss_lb_session = nn.CrossEntropyLoss()(session_outputs, session_targets.long())
                                else:
                                    suploss_lb_session = 0.0

                                if use_srd:
                                    old_mask = targets < num_old_classes
                                    tg_outputs = ref_model.get_logits(raw_feats)
                                    suploss_kd = nn.MSELoss()(tg_outputs[old_mask], ref_outputs.detach()[old_mask]) * 10
                                    suploss_reg = nn.L1Loss()(raw_feats[old_mask], ref_raw_feats.detach()[old_mask])
                                else:
                                    if kd_only_old:
                                        old_mask = targets < num_old_classes
                                        suploss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[old_mask][:, :num_old_classes] / T, dim=1),
                                                            F.softmax(ref_outputs[old_mask].detach() / T, dim=1)) * T * T * beta * num_old_classes
                                    else:
                                        suploss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
                                                                F.softmax(ref_outputs.detach() / T, dim=1)) * T * T * beta * num_old_classes
                                    suploss_reg = 0.0 

                                if use_conloss:  
                                    scores = F.cosine_similarity(text_anchor.unsqueeze(0).repeat(len(feats), 1, 1),
                                                                    feats.unsqueeze(1).repeat(1, len(text_anchor), 1), 2) / 0.1
                                    conloss_lb = F.cross_entropy(scores, targets.long()) 
                                else:
                                    conloss_lb = 0.0

                                skip = False
                                if include_unlabel:
                                    try:
                                        inputs_ulb, inputs_s_ulb, gt = next(ssl_iterator)
                                    except StopIteration:
                                        ssl_iterator = iter(ssl_trainloader)
                                        inputs_ulb, inputs_s_ulb, gt = next(ssl_iterator)
                                    
                                    inputs_ulb, inputs_s_ulb, gt = inputs_ulb.to(device), inputs_s_ulb.to(device), gt.to(device)
                                    if len(inputs_ulb) == 1:
                                        skip = True
                                        continue
                                    
                                    outputs_ulb, raw_feats_ulb, feats_ulb, session_outputs_ulb = tg_model(inputs_ulb, return_feats=True)
                                    outputs_s_ulb, raw_feats_s_ulb, feats_s_ulb, _ = tg_model(inputs_s_ulb, return_feats=True)
                                    feats_ulb, feats_s_ulb = F.normalize(feats_ulb, p=2, dim=1), F.normalize(feats_s_ulb, p=2, dim=1)
                                    
                                    if use_session_labels:
                                        session_targets = torch.ones(len(inputs_ulb))*(total_cn - nb_cl_fg) // nb_cl
                                        suploss_ulb_session = nn.CrossEntropyLoss()(session_outputs_ulb, session_targets.to(device).long())
                                    else:
                                        suploss_ulb_session = 0.0

                                    distri.append(torch.softmax(outputs_ulb[:, old_cn:total_cn], dim=-1).detach().mean(0))
                                    if len(distri) > N:
                                        distri.pop(0)
                                    
                                    if use_sim:
                                        num_ulb = len(gt)
                                        bank = mem_bank.clone().detach()
                                        with torch.no_grad():
                                            # 先验屏蔽掉旧类
                                            outputs_ulb[:, :old_cn] = -1e4
                                            outputs_ulb = F.softmax(outputs_ulb, dim=-1)
                                            teacher_logits = feats_ulb @ bank
                                            teacher_logits[:, labels_bank<old_cn] = -1e4
                                            teacher_prob_orig = F.softmax(teacher_logits / 0.5, dim=1)

                                            factor = outputs_ulb.gather(1, labels_bank.expand([num_ulb, -1]))
                                            teacher_prob = teacher_prob_orig * factor
                                            teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                                            if smoothing_alpha < 1:
                                                bs = teacher_prob_orig.size(0)
                                                aggregated_prob = torch.zeros([bs, total_cn], device=teacher_prob_orig.device)
                                                aggregated_prob = aggregated_prob.scatter_add(1, labels_bank.expand([bs,-1]) , teacher_prob_orig)
                                                probs_x_ulb_w = outputs_ulb * smoothing_alpha + aggregated_prob * (1-smoothing_alpha)
                                            else:
                                                probs_x_ulb_w = outputs_ulb

                                        student_logits = feats_s_ulb @ bank
                                        student_prob = F.softmax(student_logits / 0.5, dim=1)

                                        in_loss = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1).mean()
                                        
                                        if epoch == 0:
                                            in_loss = 0.0
                                            probs_x_ulb_w = outputs_ulb
                                        
                                        max_value, max_idx = torch.max(probs_x_ulb_w, dim=1)
                                        mask = max_value.ge(p_cutoff)    
                                        mask = mask.float()

                                        consloss_ulb = consistency_loss_raw(outputs_s_ulb, probs_x_ulb_w, 'ce', mask=mask)

                                        if use_conloss and not no_use_conloss_on_ulb:
                                            scores = F.cosine_similarity(text_anchor.unsqueeze(0).repeat(len(feats_s_ulb), 1, 1),
                                                    feats_s_ulb.unsqueeze(1).repeat(1, len(text_anchor), 1), 2) / 0.1
                                            conloss_ulb = F.cross_entropy(scores, probs_x_ulb_w.max(1)[1].long())
                                        else:
                                            conloss_ulb = 0.0

                                        update_bank(feats, targets, indexs)
                                        loss = suploss_lb + lambda_kd * suploss_kd + lambda_con * (conloss_lb + consloss_ulb) + lambda_cons * consloss_ulb + lambda_in * in_loss
               
                                    else:
                                        if use_proto:
                                            if unlabels_predict_mode == 'cosine':
                                                cosine_scores = F.cosine_similarity(prototypes_new.unsqueeze(0).repeat(len(feats_ulb), 1, 1),
                                                                                    feats_ulb.unsqueeze(1).repeat(1, len(prototypes_new), 1), 2) / 0.1
                                                pseudo_label = torch.softmax(cosine_scores, dim=1)
                                                max_probs, predicted_classes = torch.max(pseudo_label, dim=1)
                                                mask = max_probs.ge(p_cutoff).float()
                                                # predicted_classes = torch.argmax(cosine_scores, dim=1)  # (batch_size,)
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
                                            pseudo_label = torch.softmax(outputs_ulb[:, old_cn:total_cn], dim=-1)
                                            
                                            if use_da:
                                                pseudo_label = pseudo_label / distri
                                                pseudo_label = pseudo_label / pseudo_label.sum(dim=1, keepdim=True)

                                            max_probs, predicted_classes = torch.max(pseudo_label, dim=-1)
                                            mask = max_probs.ge(p_cutoff).float()
                                        
                                        predicted_classes = predicted_classes + old_cn
                                        if use_hard_labels:
                                            consloss_ulb = ce_loss(outputs_s_ulb, predicted_classes, True, reduction='none') * mask
                                        else:
                                            consloss_ulb = ce_loss(outputs_s_ulb, pseudo_label, False, reduction='none') * mask
                                        consloss_ulb = consloss_ulb.mean()
                                        
                                        ulb_total += gt.size(0)
                                        ulb_correct += predicted_classes.eq(gt).sum().item()
                                        pseudo_acc = predicted_classes.eq(gt).sum().item() / gt.size(0)
                                        
                                        if mask.bool().any():
                                            ulb_mask_total += gt[mask.bool()].size(0)
                                            ulb_mask_correct += predicted_classes[mask.bool()].eq(gt[mask.bool()]).sum().item()
                                            mask_pseudo_acc = predicted_classes[mask.bool()].eq(gt[mask.bool()]).sum().item() / gt[mask.bool()].size(0)

                                        if not mask.bool().all():
                                            no_mask_pseudo_acc = predicted_classes[torch.logical_not(mask.bool())].eq(gt[torch.logical_not(mask.bool())]).float().mean().item()
                                        
                                        if not no_use_conloss_on_ulb:
                                            # feats_ulb_masked = feats_ulb
                                            # scores = F.cosine_similarity(text_anchor.unsqueeze(0).repeat(len(feats_ulb_masked), 1, 1),
                                            # feats_ulb_masked.unsqueeze(1).repeat(1, len(text_anchor), 1), 2) / 0.1
                                            # conloss_ulb = F.cross_entropy(scores, predicted_classes.long())
                                            scores = F.cosine_similarity(text_anchor.unsqueeze(0).repeat(len(feats_ulb), 1, 1),
                                                                        feats_s_ulb.unsqueeze(1).repeat(1, len(text_anchor), 1), 2) / 0.1
                                            conloss_ulb = F.cross_entropy(scores, predicted_classes.long(), reduction='none') * mask
                                            conloss_ulb = conloss_ulb.mean()
                                        else:
                                            conloss_ulb = 0.0

                                        if use_ulb_kd:
                                            # 仿照ICCV蒸馏的无标记数据蒸馏实现
                                            _, _, ref_feats_ulb, _ = ref_model(inputs_s_ulb, return_feats=True)
                                            
                                            normalized_ref_feats_ulb = F.normalize(torch.cat((ref_feats,ref_feats_ulb)), p=2, dim=1)

                                            # 使用所有的旧类标记数据 --> similarity_all
                                            # labels_metric = F.one_hot(labels_bank[labels_bank < old_cn], num_classes=old_cn)
                                            # teacher_logits = normalized_ref_feats_ulb @ ref_mem_bank[:, labels_bank < old_cn].detach() 
                                            # teacher_prob = F.softmax(teacher_logits / 0.1, dim=1) @ labels_metric.float()
                                            # student_logits = feats_ulb @ mem_bank[:, labels_bank < old_cn].detach() 
                                            # student_prob = F.log_softmax(student_logits / 0.1, dim=1) @ labels_metric.float()
                                            # assert teacher_prob.size() == student_prob.size() 
                                            # suploss_kd_ulb = torch.sum(-teacher_prob.detach() * student_prob, dim=1).mean() * 0.1 * 0.1
                                            
                                            # 使用当前batch的旧类标记数据 --> similarity_part
                                            if old_mask.sum() > 0:
                                                # normalized_ref_feats = F.normalize(ref_raw_feats[old_mask], p=2, dim=1)
                                                # normalized_feats = F.normalize(raw_feats[old_mask], p=2, dim=1)
                                                # labels_metric = F.one_hot(targets[old_mask], num_classes=old_cn)
                                                # prototypes_ref = F.normalize(prototypes_ref, p=2, dim=1)
                                                num_prototypes = prototypes_ref.shape[0]
                                                prototype_targets = torch.arange(num_prototypes, device=prototypes_ref.device)
                                                labels_metric = F.one_hot(prototype_targets, num_classes=num_prototypes)
                                                # print(normalized_ref_feats.shape, labels_metric.shape, prototypes_old.shape)
                                                # teacher_logits = normalized_ref_feats_ulb @ normalized_ref_feats.T
                                                teacher_logits = normalized_ref_feats_ulb @ prototypes_ref.T
                                                # teacher_prob = F.softmax(teacher_logits / 0.1, dim=1) @ labels_metric.float()
                                                teacher_prob = F.softmax(teacher_logits / 0.1, dim=1)
                                                
                                                # student_logits = F.normalize(raw_feats_ulb, p=2, dim=1) @ normalized_feats.T
                                                student_logits = F.normalize(torch.cat((feats, feats_ulb)), p=2, dim=1) @ prototypes_ref.T
                                                # student_prob = F.log_softmax(student_logits / 0.1, dim=1) @ labels_metric.float()
                                                student_prob = F.log_softmax(student_logits / 0.1, dim=1)
                                                
                                                assert teacher_prob.size() == student_prob.size() 
                                                suploss_kd_ulb = torch.sum(-teacher_prob.detach() * student_prob, dim=1).mean() * 1 #* 0.2
                                            else:
                                                suploss_kd_ulb = torch.tensor(0.0).to(device)

                                        else:
                                            suploss_kd_ulb = 0.0

                                        if use_ulb_aug:
                                            p_prototypes = torch.cat([prototypes_old, prototypes_new], dim=0)
                                            # COSINE的另一种实现方式
                                            q_cosine_scores = F.cosine_similarity(p_prototypes.unsqueeze(0).repeat(len(feats_ulb), 1, 1),
                                                                                feats_ulb.unsqueeze(1).repeat(1, len(p_prototypes), 1), 2) / 0.1
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
                                                    ulb_aug_new_acc = 0.0
                                                    ulb_new_acc = 0.0
                                            
                                            consloss_ulb_aug = ce_loss(outputs_s_ulb, q_predict_class, True, reduction='none') * torch.logical_not(mask.bool()).float()
                                                                                        
                                            consloss_ulb_aug = consloss_ulb_aug.mean()
                                        else:
                                            consloss_ulb_aug = 0.0
                                        
                                        loss = suploss_lb \
                                                + lambda_kd * (suploss_kd + suploss_kd_ulb) \
                                                + lambda_session * (suploss_lb_session + suploss_ulb_session) \
                                                + lambda_con * (conloss_lb + conloss_ulb) \
                                                + lambda_reg * suploss_reg \
                                                + lambda_cons * (consloss_ulb + consloss_ulb_aug) \
                                
                                else:
                                    loss = suploss_lb \
                                            + lambda_kd * suploss_kd \
                                            + lambda_reg * suploss_reg \
                                            + lambda_session * suploss_lb_session \
                                            + lambda_con * conloss_lb

                                loss.backward()
                                tg_optimizer.step()
                                # tg_lr_scheduler.step()

                                train_loss += loss.item()
                                train_suploss_kd += suploss_kd.item()
                                train_suploss_lb += suploss_lb.item()
                                train_suploss_reg += suploss_reg.item() if use_srd else 0.0
                                train_suploss_session += suploss_lb_session.item() if use_session_labels else 0.0
                                train_conloss_lb += conloss_lb.item() if use_conloss else 0.0
                                train_consloss_ulb += consloss_ulb.item()  if include_unlabel else 0.0
                                train_suploss_ulb_session += suploss_ulb_session.item() if include_unlabel and use_session_labels else 0.0
                                train_conloss_ulb += conloss_ulb.item() if include_unlabel and not no_use_conloss_on_ulb else 0.0
                                train_suploss_kd_ulb += suploss_kd_ulb.item() if include_unlabel and use_ulb_kd else 0.0
                                train_inloss_ulb += in_loss.item() if include_unlabel and use_sim and epoch!=0 else 0.0
                                train_consloss_ulb_aug += consloss_ulb_aug.item() if include_unlabel and use_ulb_aug else 0.0
                                train_util_ratio += mask.mean().item() if include_unlabel else 0.0  
            
                                _, predicted = outputs.max(1)
                                total += targets.size(0)
                                correct += predicted.eq(targets).sum().item()

                                writer.add_scalar('Unlabel_Train_Stage/SupLoss_LB', suploss_lb.item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)
                                writer.add_scalar('Unlabel_Train_Stage/SupLoss_KD', suploss_kd.item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)
                                
                                if use_conloss:
                                    writer.add_scalar('Unlabel_Train_Stage/ConLoss_LB', conloss_lb.item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)
                                
                                if use_srd:
                                    writer.add_scalar('Unlabel_Train_Stage/SupLoss_REG', suploss_reg.item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)

                                if use_session_labels:
                                    writer.add_scalar('Unlabel_Train_Stage/SupLoss_SESSION', suploss_lb_session.item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)

                                if include_unlabel and not skip:
                                    writer.add_scalar('Unlabel_Train_Stage/ConsLoss_ULB', consloss_ulb.item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)
                                    writer.add_scalar('Unlabel_Train_Stage/Ratio_ULB', mask.mean().item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)
                                    writer.add_scalars('Unlabel_Train_Stage/Pseusdo_Acc', {'mask_acc': mask_pseudo_acc if mask.bool().any() else 0.0, 'no_mask_acc': no_mask_pseudo_acc if not mask.bool().all() else 0.0, 'acc': pseudo_acc}, (u_i*10 + epoch) * len(trainloader_1) + batch_idx)
                
                                    if not no_use_conloss_on_ulb:
                                        writer.add_scalar('Unlabel_Train_Stage/ConLoss_ULB', conloss_ulb.item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)
                                    if use_ulb_kd:
                                        writer.add_scalar('Unlabel_Train_Stage/SupLoss_KD_ULB', suploss_kd_ulb.item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)
                                    if use_sim:
                                        writer.add_scalar('Unlabel_Train_Stage/InLoss_ULB', in_loss.item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)    
                                    if use_session_labels:
                                        writer.add_scalar('Unlabel_Train_Stage/SupLoss_ULB_SESSION', suploss_ulb_session.item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)
                                    if use_ulb_aug:
                                        writer.add_scalar('Unlabel_Train_Stage/ConsLoss_ULB_AUG', consloss_ulb_aug.item(), (u_i*10 + epoch) * len(trainloader_1) + batch_idx)
                                        if not mask.bool().all():
                                            writer.add_scalars('Unlabel_Train_Stage/Ulb_Class_Num', {'old': old_class_num, 'new': new_class_num}, (u_i*10 + epoch) * len(trainloader_1) + batch_idx) 
                                            writer.add_scalars('Unlabel_Train_Stage/Acc_ULB_Aug', {'proto':  ulb_aug_new_acc, 'classifier': ulb_new_acc}, (u_i*10 + epoch) * len(trainloader_1) + batch_idx) 


                            tg_lr_scheduler.step()
                            
                            if update_proto:
                                prototypes_old, prototypes_new, pro = get_proto(trainloader, tg_model, old_cn, device)
                            
                            test_loss, test_acc, test_loss_session, test_acc_session = validate(tg_model, testloader, device, weight_per_class, nb_cl_fg, nb_cl)
                            pseudo_label_acc_p, pseudo_label_acc_s = test_pseudo_acc(tg_model, prototypes_new, ssl_trainloader, unlabels_predict_mode,old_cn, total_cn, device)

                            writer.add_scalars("Unlabels Training Stage Accuracy", {"Train": 100.*correct/total, "Test": test_acc}, u_i*10+epoch)
                            writer.add_scalars("Unlabels Training Stage  Loss", {"Train": train_loss / (batch_idx + 1), "Test": test_loss}, u_i*10+epoch)
                            writer.add_scalars("Unlabels Training Stage  Pseudo Accuracy", {"Acc_p": pseudo_label_acc_p, "Acc_s": pseudo_label_acc_s}, u_i*10+epoch)
                            
                            if use_session_labels:
                                writer.add_scalar("Unlabels Training Stage Session Accuracy", test_acc_session, u_i*10+epoch)
                                writer.add_scalar("Unlabels Training Stage Session Loss",  test_loss_session, u_i*10+epoch)

                            if epoch % 2 == 1:
                                print('Epoch: {}, acc_classifier: {}, acc_proto: {}'.format(epoch, pseudo_label_acc_p, pseudo_label_acc_s))
                                print('Epoch: {}, Trainset: {}, Unlabel_trainset: {}, Lr: {}, P_cutoff: {}, Lambda_KD: {}, Lambda_CON: {}, Lambda_CONS: {}, Lambda_SESSION: {}'.format(epoch, len(trainset_1), \
                                        len(ssl_trainloader.dataset), tg_lr_scheduler.get_last_lr()[0], p_cutoff, lambda_kd, lambda_con, lambda_cons, lambda_session))
                                print('Epoch: {}, SupLoss_LB: {:.4f}, SupLoss_KD: {:.4f}, SupLoss_REG: {:.4f}, SupLoss_SESSION: {:.4f}, ConLoss_LB: {:.4f}, ConsLoss_ULB: {:.4f}, ConsLoss_ULB_Aug: {:.4f}, SupLoss_KD_ULB: {:.4f}, ConLoss_ULB: {:.4f}, SupLoss_SESSION_ULB: {:.4f}, InLoss_ULB: {:.4f}, Loss: {:.4f} Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Test Seesion Loss: {:.4f}, Test Session Acc: {:.4f}'.format(epoch, \
                                        train_suploss_lb / (batch_idx+1), train_suploss_kd / (batch_idx+1), train_suploss_reg /(batch_idx+1), train_suploss_session / (batch_idx+1), train_conloss_lb / (batch_idx+1), train_consloss_ulb / (batch_idx+1), train_consloss_ulb_aug / (batch_idx+1), train_suploss_kd_ulb / (batch_idx+1), 
                                        train_conloss_ulb / (batch_idx+1), train_suploss_ulb_session / (batch_idx+1), train_inloss_ulb / (batch_idx+1), train_loss / (batch_idx+1), 100. * correct / total, test_loss, test_acc, test_loss_session, test_acc_session))
                                
                        
                    if unlabeled_data.shape[0] < 1:
                        unlabeled_data = None
                    else:
                        unlabeled_trainset = UnlabelDataset(image_size, dataset=args.dataset, autoaug=args.autoaug)
                        unlabeled_trainset.data = unlabeled_data
                        unlabeled_trainset.targets = unlabeled_gt
                        unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=train_batch_size,
                                                                            shuffle=False, num_workers=4)
            
            print("selected_unlabeld_data: ", selected_unlabeld_data.shape)
            print("selected_unlabeld_targets: ", selected_unlabeld_targets.shape)
            print("selected_unlabeld_predicts: ", selected_unlabeld_predicts.shape)
            print("selected data acc: ", (selected_unlabeld_predicts == selected_unlabeld_targets).sum() / len(selected_unlabeld_targets))
            print("selected data true distribution: ", np.unique(selected_unlabeld_targets, return_counts=True))
            print("selected data predicts distribution: ", np.unique(selected_unlabeld_predicts, return_counts=True))
            
        else:
            raise ValueError('method: {} not supported'.format(method))
        
    # eval
    loss, acc, loss_session, acc_session = validate(tg_model, testloader, device, weight_per_class, nb_cl_fg, nb_cl)
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
        assert logits.shape == targets.shape
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
                max_probs, predicted_classes = torch.max(pseudo_label, dim=1)
                mask = max_probs.ge(p_cutoff).float()
                # predicted_classes = torch.argmax(cosine_scores, dim=1)  # (batch_size,)
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

            max_probs, predicted_classes = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff).float()
            indices = mask.nonzero(as_tuple=True)[0]
            # predicted_classes = max_idx
        
        predicted_classes = predicted_classes + old_cn
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, predicted_classes, use_hard_labels, reduction='none') * mask
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


def validate(tg_model, testloader, device, weight_per_class, nb_cl_fg=None, nb_cl=None):
    # eval
    tg_model.eval()
    test_loss = 0
    test_loss_session = 0
    correct = 0
    correct_session = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _, session_outputs = tg_model(inputs, return_feats=True)
            loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return test_loss/(batch_idx+1), 100.*correct/total, test_loss_session/(batch_idx+1), 100.*correct_session/total

def get_session_labels(class_labels: torch.tensor, nb_cl_fg: int, nb_cl: int):
    session_labels = torch.ones_like(class_labels) * -1
    for i in range(len(class_labels)):
        if class_labels[i] < nb_cl_fg:
            session_labels[i] = 0
        else:
            session_labels[i] = (class_labels[i] - nb_cl_fg) // (nb_cl) + 1
    return session_labels.to(class_labels)


def test_pseudo_acc(tg_model, prototypes_new, unlabeled_trainloader, unlabels_predict_mode, old_cn, total_cn, device):
    acc_p = 0
    acc_s = 0
    totalnum_p = 0

    for batch_idx, (inputs, _, gt) in enumerate(unlabeled_trainloader):
        inputs = inputs.to(device)
        gt = gt.to(device)
        
        if len(inputs) == 1:
            continue    
        
        outputs, raw_feats, feats, session_outputs = tg_model(inputs, return_feats=True)
        outputs_new = F.softmax(outputs[:, old_cn: total_cn], dim=1)
        feats = F.normalize(feats, p=2, dim=1)

        max_value, max_idx = torch.max(outputs_new, dim=1)
        acc_p += (max_idx+old_cn).eq(gt).sum().cpu().item()  
        totalnum_p += len(gt)
        
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
            outputs_new_proto = torch.softmax(cosine_scores, dim=1)  
            max_value_proto, max_idx_proto = torch.max(outputs_new_proto, dim=1)
            predicted_classes = torch.argmax(cosine_scores, dim=1) 
        else:
            raise ValueError('unlabels_predict_mode: {} not supported'.format(unlabels_predict_mode))
        
        predicted_classes = predicted_classes + old_cn
        acc_s += predicted_classes.eq(gt).sum().cpu().item()  

    return 100*acc_p/totalnum_p, 100*acc_s/totalnum_p