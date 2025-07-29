# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning import distances
from torch import Tensor

class ContrastiveLossTorch:

    def __init__(self, threshold: float, hard: Optional[bool] = None):
        self.threshold = threshold
        self.hard = hard if hard is not None else False

    def build_loss_matrix(self, embs: Tensor, ys: Tensor):
        lpembdist = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
        emb_distance_matrix = lpembdist(embs)

        lpydist = distances.LpDistance(normalize_embeddings=False, p=1, power=1)
        y_distance_matrix = lpydist(ys)

        loss = torch.zeros_like(emb_distance_matrix).to(embs)

        threshold_matrix = self.threshold * torch.ones(loss.shape).to(embs)

        high_dy_filter = y_distance_matrix > self.threshold
        aux_max_dz_thr = torch.maximum(emb_distance_matrix, threshold_matrix)
        aux_min_dz_thr = torch.minimum(emb_distance_matrix, threshold_matrix)

        if self.hard:
            # dy - dz
            loss[high_dy_filter] = y_distance_matrix[high_dy_filter] - emb_distance_matrix[high_dy_filter]
            # dz
            loss[~high_dy_filter] = emb_distance_matrix[~high_dy_filter]
        else:
            # (2 - min(threshold, dz) / threshold) * (dy - max(dz, threshold))
            loss[high_dy_filter] = (2 - aux_min_dz_thr[high_dy_filter]).div(self.threshold) * (
                    y_distance_matrix[high_dy_filter] - aux_max_dz_thr[high_dy_filter])

            #  max(threshold, dz) / threshold * (min(dz, threshold) - dy)
            loss[~high_dy_filter] = aux_max_dz_thr[~high_dy_filter].div(self.threshold) * (
                    aux_min_dz_thr[~high_dy_filter] - y_distance_matrix[~high_dy_filter])

        loss = torch.relu(loss)
        return loss

    def compute_loss(self, embs: Tensor, ys: Tensor):
        loss_matrix = torch.triu(self.build_loss_matrix(embs, ys), diagonal=1)
        n = (loss_matrix > 0).sum()

        if n == 0:
            n = 1
        # average over non-zero elements
        return loss_matrix.sum().div(n)

    def __call__(self, embs: Tensor, ys: Tensor):
        return self.compute_loss(embs, ys)

    @staticmethod
    def exp_metric_id(threshold: float, hard: Optional[bool] = None) -> str:
        metric_id = f'contrast-thr-{threshold:g}'
        if hard:
            metric_id += '-hard'
        return metric_id


class TripletLossTorch:
    def __init__(self, threshold: float, margin: Optional[float] = None, soft: Optional[bool] = False,
                 eta: Optional[float] = None):
        """
        Compute Triplet loss
        Args:
            threshold: separate positive and negative elements in temrs of `y` distance
            margin: hard triplet loss parameter
            soft: whether to use sigmoid version of triplet loss
            eta: parameter of hyperbolic function softening transition between positive and negative classes
        """
        self.threshold = threshold
        self.margin = margin
        self.soft = soft
        assert eta is None or eta > 0, eta
        self.eta = eta

    def build_loss_matrix(self, embs: Tensor, ys: Tensor):
        lpembdist = distances.LpDistance(normalize_embeddings=False, p=2, power=1)
        emb_distance_matrix = lpembdist(embs)

        lpydist = distances.LpDistance(normalize_embeddings=False, p=1, power=1)
        y_distance_matrix = lpydist(ys)

        positive_embs = emb_distance_matrix.where(y_distance_matrix <= self.threshold, torch.tensor(0.).to(embs))
        negative_embs = emb_distance_matrix.where(y_distance_matrix > self.threshold, torch.tensor(0.).to(embs))

        loss_loop = 0 * torch.tensor([0.], requires_grad=True).to(embs)
        n_positive_triplets = 0
        for i in range(embs.size(0)):
            pos_i = positive_embs[i][positive_embs[i] > 0]
            neg_i = negative_embs[i][negative_embs[i] > 0]
            pairs = torch.cartesian_prod(pos_i, -neg_i)
            if self.soft:
                triplet_losses_for_anchor_i = torch.nn.functional.softplus(pairs.sum(dim=-1))
                if self.eta is not None:
                    # get the corresponding delta ys
                    pos_y_i = y_distance_matrix[i][positive_embs[i] > 0]
                    neg_y_i = y_distance_matrix[i][negative_embs[i] > 0]
                    pairs_y = torch.cartesian_prod(pos_y_i, neg_y_i)
                    assert pairs.shape == pairs_y.shape, (pairs_y.shape, pairs.shape)
                    triplet_losses_for_anchor_i = triplet_losses_for_anchor_i * \
                                                  self.smooth_indicator(self.threshold - pairs_y[:, 0]) \
                                                      .div(self.smooth_indicator(self.threshold)) \
                                                  * self.smooth_indicator(pairs_y[:, 1] - self.threshold) \
                                                      .div(self.smooth_indicator(1 - self.threshold))
            else:
                triplet_losses_for_anchor_i = torch.relu(self.margin + pairs.sum(dim=-1))
            n_positive_triplets += (triplet_losses_for_anchor_i > 0).sum()
            loss_loop += triplet_losses_for_anchor_i.sum()
        loss_loop = loss_loop.div(max(1, n_positive_triplets))

        return loss_loop

    def smooth_indicator(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, float):
            return np.tanh(x / (2 * self.eta))
        return torch.tanh(x / (2 * self.eta))

    def compute_loss(self, embs: Tensor, ys: Tensor):
        return self.build_loss_matrix(embs, ys)

    def __call__(self, embs: Tensor, ys: Tensor):
        return self.compute_loss(embs, ys)

    @staticmethod
    def exp_metric_id(threshold: float, margin: Optional[float] = None, soft: Optional[bool] = None,
                      eta: Optional[bool] = None) -> str:
        metric_id_base = f'triplet-thr-{threshold:g}'
        if margin is not None:
            return f'{metric_id_base}-mrg-{margin:g}'
        if soft is not None:
            metric_id = f'{metric_id_base}-soft'
            if eta is not None:
                metric_id += f'-eta-{eta:g}'
            return metric_id
        else:
            return metric_id_base


class LogRatioLossTorch:
    def __init__(self):
        """
        Compute Log-ration loss (https://arxiv.org/pdf/1904.09626.pdf)
        """
        pass

    def build_loss_matrix(self, embs: Tensor, ys: Tensor):
        eps = 1e-4 / embs.size(0)

        lpembdist = distances.LpDistance(normalize_embeddings=False, p=2, power=2)
        emb_distance_matrix = torch.sqrt(lpembdist(embs) + eps)  # L2dist

        lpydist = distances.LpDistance(normalize_embeddings=False, p=1, power=1)
        y_distance_matrix = lpydist(ys)

        eps = 1e-6

        loss_loop = 0 * torch.tensor([0.], requires_grad=True).to(embs)
        n_positive_triplets = 0
        m = embs.size()[0] - 1  # #paired

        for ind_a in range(embs.size(0)):
            # auxiliary variables
            idxs = torch.arange(0, m).to(device=embs.device)
            idxs[ind_a:] += 1

            log_dist = torch.log(emb_distance_matrix[ind_a][idxs] + eps)
            log_y_dist = torch.log(y_distance_matrix[ind_a][idxs] + eps)

            diff_log_dist = log_dist.repeat(m, 1).t() - log_dist.repeat(m, 1)
            diff_log_y_dist = log_y_dist.repeat(m, 1).t() - log_y_dist.repeat(m, 1)
            assert diff_log_y_dist.shape == diff_log_dist.shape == (m, m), (diff_log_y_dist.shape,
                                                                            diff_log_dist.shape, m)
            valid_aij = diff_log_y_dist < 0  # keep triplet having D(y_a, y_i) < D(y_q, y_j)

            log_ratio_loss = (diff_log_dist - diff_log_y_dist).pow(2)[valid_aij].sum()

            loss_loop += log_ratio_loss
            n_positive_triplets += valid_aij.sum()

        loss_loop = loss_loop.div(max(1, n_positive_triplets))

        return loss_loop

    def compute_loss(self, embs: Tensor, ys: Tensor):
        return self.build_loss_matrix(embs, ys)

    def __call__(self, embs: Tensor, ys: Tensor):
        return self.compute_loss(embs, ys)

    @staticmethod
    def exp_metric_id() -> str:
        metric_id = "log-ratio"
        return metric_id


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, device=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if device is None:
            device = (torch.device('cuda')
                    if features.is_cuda
                    else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class Required:
    def __init__(self):
        pass


class NotRequired:
    def __init__(self):
        pass


METRIC_LOSSES = {
    'triplet': {
        'kwargs': {'threshold': Required(),
                   'margin': None,
                   'soft': None,
                   'eta': None
                   },
        'exp_metric_id': TripletLossTorch.exp_metric_id
    },
    'log_ratio': {
        'kwargs': {},
        'exp_metric_id': LogRatioLossTorch.exp_metric_id
    }
}