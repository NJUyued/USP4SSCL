#!/usr/bin/env python
# coding=utf-8
import pdb
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
from sklearn.metrics import confusion_matrix
from utils_pytorch import *

def compute_accuracy(tg_model, tg_feature_model, class_means, evalloader, text_anchor=None, use_text_anchor=False, scale=None, print_info=True, 
                     session_means=None, start_session=None, nb_cl=None, device=None):
    if device is None:
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.cuda.current_device()
    tg_model.eval()
    tg_feature_model.eval()

    correct = 0
    correct_uad = 0
    correct_etf = 0
    correct_test = 0
    correct_session = 0
    correct_composite = 0
    correct_composite_etf = 0
    
    total = 0
    total_m = 0

    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            
            outputs, _, feats, _ = tg_model(inputs, return_feats=True)
            # 原型更改
            # outputs, feats, _, _ = tg_model(inputs, return_feats=True)
            
            outputs = F.softmax(outputs, dim=1)

            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            conf, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_feature = tg_feature_model(inputs).data

            # Compute score for UaD-CE
            sqd_uad = cdist(class_means[:,:,0].T, np.squeeze(outputs_feature.cpu().numpy()), 'sqeuclidean')
            score_uad = torch.from_numpy((-sqd_uad).T).to(device)
            _, predicted_uad = score_uad.max(1)
            correct_uad += predicted_uad.eq(targets).sum().item()

            # 计算复合acc
            for i in range(targets.size(0)):
                if conf[i] > 0.95:  # 当置信度大于0.95时使用cnn_acc
                    if predicted[i] == targets[i]:
                        correct_composite += 1
                else:  # 当置信度小于0.95时使用uad_acc的预测
                    if predicted_uad[i] == targets[i]:
                        correct_composite += 1

            if session_means is not None:
                # Compute score for ETF
                sqd_session = cdist(session_means.T, np.squeeze(feats.cpu().numpy()), 'sqeuclidean')
                score_session = torch.from_numpy((-sqd_session).T).to(device)
                _, predicted_session = score_session.max(1)
                targets_session = torch.tensor([(l - start_session*nb_cl)//10 if l > start_session*nb_cl else 0 for l in targets]).long().to(device)
                correct_session += predicted_session.eq(targets_session).sum().item()
                
                start_class = [(session if session == 0 else (session + start_session) * nb_cl) for session in predicted_session]
                end_class = [(session + start_session + 1) * nb_cl for session in predicted_session]
                
                predicted_test = torch.ones(targets.size(0)).to(device)*-1
                for i in range(len(targets)):
                    predicted_test[i] = score_uad[i][start_class[i]:end_class[i]].argmax() + start_class[i]
                
                correct_test += predicted_test.eq(targets).sum().item()
            
            if use_text_anchor: 
                sqd_etf = cdist(text_anchor.T, np.squeeze(feats.cpu().numpy()), 'sqeuclidean')
                score_etf = torch.from_numpy((-sqd_etf).T).to(device)
                _, predicted_classes_etf = score_etf.max(1)
                correct_etf += predicted_classes_etf.eq(targets).sum().item()
            
                # feats = F.normalize(feats, p=2, dim=1)
                # # cosine_scores = F.linear(feats, torch.from_numpy(text_anchor).to(feats).T) / 0.1 # COSINE的另一种实现方式
                # cosine_scores = F.cosine_similarity(torch.from_numpy(text_anchor).to(feats).T.unsqueeze(0).repeat(len(feats), 1, 1),
                #                                     feats.unsqueeze(1).repeat(1, len(text_anchor.T), 1), 2) / 0.1
                # pseudo_label = torch.softmax(cosine_scores, dim=1)
                # max_probs, predicted_classes_etf = torch.max(pseudo_label, dim=1)
                # # class_means_squared = torch.sum(text_anchor**2, dim=1, keepdim=True)  # (num_classes, 1)
                # # outputs_feature_squared = torch.sum(feats**2, dim=1, keepdim=True).T  # (1, batch_size)
                # # dot_product = torch.matmul(text_anchor, feats.T)  # (num_classes, batch_size)
                # # squared_distances = class_means_squared + outputs_feature_squared - 2 * dot_product  # (num_classes, batch_size)
                # # predicted_classes = torch.argmin(squared_distances, dim=0)  # (batch_size,)
                # correct_etf += predicted_classes_etf.eq(targets).sum().item() 
                # 计算复合acc
                for i in range(targets.size(0)):
                    if conf[i] > 0.95:  # 当置信度大于0.95时使用cnn_acc
                        total_m += 1
                        if predicted[i] == targets[i]:
                            correct_composite_etf += 1
                    else:  # 当置信度小于0.95时使用uad_acc的预测
                        if predicted_classes_etf[i] == targets[i]:
                            correct_composite_etf += 1

    cnn_acc = 100.*correct/total
    uad_acc = 100.*correct_uad/total
    etf_acc = 100.*correct_etf/total
    composite_acc = 100. * correct_composite / total
    composite_etf_acc = 100. * correct_composite_etf / total
    test_acc = 100.*correct_test/total
    session_acc = 100.*correct_session/total

    if print_info:
        print("upper bound 0.95:\t\t{} / {} ".format(total_m, total))
        print("  top 1 accuracy UaD-CE          :\t\t{:.2f} % / {:.2f} % / {:.2f} % / {:.2f} %/ {:.2f}%".format(cnn_acc, uad_acc, composite_acc, etf_acc, composite_etf_acc))

    # return [cnn_acc, uad_acc]
    return uad_acc

def compute_accuracy_train(tg_model, tg_feature_model, class_means, evalloader, text_anchor=None, scale=None, print_info=True, 
                     session_means=None, start_session=None, nb_cl=None, device=None):
    if device is None:
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.cuda.current_device()
    tg_model.eval()
    tg_feature_model.eval()

    correct = 0
    correct_uad = 0
    correct_etf = 0
    correct_test = 0
    correct_session = 0

    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            
            outputs, _, feats, _ = tg_model(inputs, return_feats=True)
            # 原型更改
            # outputs, feats, _, _ = tg_model(inputs, return_feats=True)
            
            outputs = F.softmax(outputs, dim=1)

            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_feature = tg_feature_model(inputs).data

            # Compute score for UaD-CE
            # sqd_uad = cdist(class_means[:,:,0].T, np.squeeze(outputs_feature.cpu().numpy()), 'sqeuclidean')
            sqd_uad = torch.cdist(class_means, feats, p=2) ** 2
            score_uad = -sqd_uad.T
            _, predicted_uad = score_uad.max(1)
            correct_uad += predicted_uad.eq(targets).sum().item()

            if session_means is not None:
                # Compute score for ETF
                sqd_session = cdist(session_means.T, np.squeeze(feats.cpu().numpy()), 'sqeuclidean')
                score_session = torch.from_numpy((-sqd_session).T).to(device)
                _, predicted_session = score_session.max(1)
                targets_session = torch.tensor([(l - start_session*nb_cl)//10 if l > start_session*nb_cl else 0 for l in targets]).long().to(device)
                correct_session += predicted_session.eq(targets_session).sum().item()
                
                start_class = [(session if session == 0 else (session + start_session) * nb_cl) for session in predicted_session]
                end_class = [(session + start_session + 1) * nb_cl for session in predicted_session]
                
                predicted_test = torch.ones(targets.size(0)).to(device)*-1
                for i in range(len(targets)):
                    predicted_test[i] = score_uad[i][start_class[i]:end_class[i]].argmax() + start_class[i]
                
                correct_test += predicted_test.eq(targets).sum().item()
            
            # outputs_feature = outputs_feature.squeeze()
            # feats = F.normalize(feats.squeeze(), p=2, dim=0)
            # class_means_squared = torch.sum(text_anchor**2, dim=1, keepdim=True)  # (num_classes, 1)
            # outputs_feature_squared = torch.sum(feats**2, dim=1, keepdim=True).T  # (1, batch_size)
            # dot_product = torch.matmul(text_anchor, feats.T)  # (num_classes, batch_size)
            # squared_distances = class_means_squared + outputs_feature_squared - 2 * dot_product  # (num_classes, batch_size)
            # predicted_classes = torch.argmin(squared_distances, dim=0)  # (batch_size,)
            # correct_etf += predicted_classes.eq(targets).sum().item() 

    cnn_acc = 100.*correct/total
    uad_acc = 100.*correct_uad/total
    # etf_acc = 100.*correct_etf/total
    test_acc = 100.*correct_test/total
    session_acc = 100.*correct_session/total

    if print_info:
        print("  top 1 accuracy UaD-CE          :\t\t{:.2f} % / {:.2f} % / {:.2f} % / {:.2f} %".format(cnn_acc, uad_acc, test_acc, session_acc))

    # return [cnn_acc, uad_acc]
    return cnn_acc                         

def compute_accuracy_t(tg_model, evalloader, text_anchor, scale=None, print_info=True, device=None):
    if device is None:
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.cuda.current_device()
    tg_model.eval()

    correct = 0
    correct_uad = 0
    correct_etf = 0
    correct_test = 0
    correct_session = 0

    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            
            outputs, _, feats, _ = tg_model(inputs, return_feats=True)
            
            outputs = F.softmax(outputs, dim=1)

            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        
            feats = F.normalize(feats.squeeze(), p=2, dim=0)
            class_means_squared = torch.sum(text_anchor**2, dim=1, keepdim=True)  # (num_classes, 1)
            outputs_feature_squared = torch.sum(feats**2, dim=1, keepdim=True).T  # (1, batch_size)
            dot_product = torch.matmul(text_anchor, feats.T)  # (num_classes, batch_size)
            squared_distances = class_means_squared + outputs_feature_squared - 2 * dot_product  # (num_classes, batch_size)
            predicted_classes = torch.argmin(squared_distances, dim=0)  # (batch_size,)
            correct_etf += predicted_classes.eq(targets).sum().item() 

    cnn_acc = 100.*correct/total
    uad_acc = 100.*correct_uad/total
    etf_acc = 100.*correct_etf/total
    test_acc = 100.*correct_test/total
    session_acc = 100.*correct_session/total

    if print_info:
        print("  top 1 accuracy UaD-CE:\t\t {:.2f} % / {:.2f} % / {:.2f} % / {:.2f} % / {:.2f} %".format(cnn_acc, uad_acc, etf_acc, test_acc, session_acc))

    return [cnn_acc, uad_acc]
                             