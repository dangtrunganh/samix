# -*- coding:  utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
from src.utils.general_utils import print_x
from src.plas_losses.base_loss import BaseLoss
from src.plas_losses.base_prototype_loss import BasePrototypeBasedLoss
import argparse


class FNC2(BaseLoss, BasePrototypeBasedLoss, nn.Module):
    def set_set_prototypes(self, set_prototypes):
        super().set_set_prototypes(set_prototypes)

    def __init__(self,
                 temperature: float = 0.07,
                 base_temperature: float = 0.07,
                 focal_gamma: int = 0,
                 **kwargs):
        super(FNC2, self).__init__()
        print_x(f"**kwargs = {kwargs}")
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.focal_gamma = focal_gamma
        print(self)

    def __str__(self):
        lines = [f"temperature: {self.temperature}", f"base_temperature: {self.base_temperature}"]
        return "[FNC2]" + "\n" + "\n".join(["\t" + line for line in lines])

    def get_mask(self, labels, mask, batch_size: int, device):
        # labels|  mask | mask_output
        #   0   |   0   | torch.eye(batch_size)
        #   0   |   1   | mask
        #   1   |   0   | torch.eq(labels, labels.T)
        #   1   |   1   | "Cannot define both `labels` and `mask`"
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            return torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            return torch.eq(labels, labels.T).float().to(device)
        else:
            return mask.float().to(device)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser):
        print_x(f"Adding specific args for FNC2....")
        parser.add_argument('--focal_gamma', type=int, default=0)
        parser.add_argument('--current_temp_sprd', type=float, default=None,
                            help='temperature for current - S-PRD distillation')
        parser.add_argument('--past_temp_sprd', type=float, default=None,
                            help='temperature for previous - S-PRD distillation')

    def compute_fnc2_loss(self, anchor_feature, anchor_count, labels, batch_size, device, target_labels,
                          corresponding_prototypes):
        """
        @param anchor_feature: [bsz x 2, d]
        @param anchor_count: 2
        @param labels:
        @param batch_size:
        @param device:
        @param target_labels:
        @param corresponding_prototypes:
        @return:
        """
        mask = self.get_mask(labels, None, batch_size, device)
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), self.temperature)
        index_previous_proto = min(target_labels)
        previous_proto = self.points[:index_previous_proto]
        anchor_contrast_previous_proto = None
        if previous_proto.numel() != 0:
            anchor_contrast_previous_proto = torch.div(
                torch.matmul(anchor_feature, previous_proto.T), self.temperature)

        points_dot_contrast = torch.div(
            torch.mul(anchor_feature, corresponding_prototypes).sum(1), self.temperature
        )
        I = torch.arange(corresponding_prototypes.size(0))
        anchor_dot_contrast[I, I] = points_dot_contrast

        if anchor_contrast_previous_proto is not None:
            logits_max, _ = torch.max(torch.cat((anchor_dot_contrast, anchor_contrast_previous_proto), dim=1), dim=1,
                                      keepdim=True)
            previous_logits_sample_proto = anchor_contrast_previous_proto - logits_max.detach()
        else:
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        exp_logits = torch.exp(logits) * logits_mask
        if anchor_contrast_previous_proto is not None:
            exp_previous_proto = torch.exp(previous_logits_sample_proto)
            log_prob = logits - torch.log(
                exp_logits.sum(1, keepdim=True) + exp_previous_proto.sum(1, keepdim=True))
        else:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        weight = (1 - torch.exp(log_prob)) ** self.focal_gamma
        pos_per_sample = mask.sum(1)
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (weight * mask * log_prob).sum(1) / pos_per_sample
        mean_log_prob_pos = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        print_x(f"Shape of mean_log_prob_pos in FNC2 loss = {mean_log_prob_pos.shape}")
        return mean_log_prob_pos  # [2 x bsz,]

    def forward(self, features, labels=None, target_labels=None, **kwargs):
        """
        @param features: [bsz, 2, d]
        @param labels: [bsz,]
        @param target_labels: [n_clas_per_task]
        @param kwargs:
        @return:
        """
        assert target_labels is not None and len(
            target_labels) > 0, "Target labels should be given as a list of integer"
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, d-dimensions],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        anchor_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        batch_size, anchor_count = features.shape[:2]
        self.points = self.points.to(features)
        repeated_labels = labels.repeat(anchor_count)
        corresponding_prototypes = self.points[repeated_labels]
        mean_log_prob_pos = self.compute_fnc2_loss(anchor_feature=anchor_feature, anchor_count=anchor_count,
                                                   labels=labels, batch_size=batch_size, device=device,
                                                   target_labels=target_labels,
                                                   corresponding_prototypes=corresponding_prototypes)
        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels:
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        loss = curr_class_mask * mean_log_prob_pos.view(anchor_count, batch_size)
        return loss.mean()
