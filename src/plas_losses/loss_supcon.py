from __future__ import print_function
import torch
import torch.nn as nn
from src.utils.general_utils import print_x
from src.plas_losses.base_loss import BaseLoss

import argparse

'''
The original source code from Co2L[ICCV'21] paper
'''


class SupConLoss(BaseLoss, nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, **kwargs):
        super(SupConLoss, self).__init__()
        print_x(f"**kwargs = {kwargs}")
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser):
        pass

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean', **kwargs):
        assert target_labels is not None and len(
            target_labels) > 0, "Target labels should be given as a list of integer"
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, d-dimensions],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]  # [bsz, 2, d]
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
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        pos_per_sample = mask.sum(1)
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels:
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        loss = curr_class_mask * loss.view(anchor_count, batch_size)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))
        return loss
