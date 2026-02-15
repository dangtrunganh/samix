# -*- coding:  utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
from src.plas_losses.base_prototype_loss import BasePrototypeBasedLoss
from src.utils.general_utils import print_x
from src.plas_losses.base_loss import BaseLoss
import argparse


class NCSAMix(BaseLoss, BasePrototypeBasedLoss, nn.Module):
    def set_set_prototypes(self, set_prototypes):
        super().set_set_prototypes(set_prototypes)

    def __init__(self,
                 loss_normal_samples: str = None,
                 temperature: float = 0.07,
                 base_temperature: float = 0.07,
                 focal_gamma: int = 0,
                 weight_samix_loss: float = 1.0,
                 **kwargs):
        nn.Module.__init__(self)
        BasePrototypeBasedLoss.__init__(self)
        BaseLoss.__init__(self)
        print_x(f"**kwargs = {kwargs}")
        self.loss_normal_samples = loss_normal_samples
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.focal_gamma = focal_gamma
        self.weight_samix_loss = weight_samix_loss
        self.print_one_previous = True
        self.print_one_future = True
        print(self)

    def __str__(self):
        lines = [f"loss_normal_samples = {self.loss_normal_samples}",
                 f"weight_samix_loss = {self.weight_samix_loss}",
                 f"temperature: {self.temperature}",
                 f"base_temperature: {self.base_temperature}",
                 f"focal_gamma: {self.focal_gamma}"]
        return "[NCSAMix]" + "\n" + "\n".join(["\t" + line for line in lines])

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser):
        print_x(f"Adding specific args....")
        parser.add_argument('--focal_gamma', type=int, default=None)
        parser.add_argument('--current_temp_sprd', type=float, default=None,
                            help='temperature for current - S-PRD distillation')
        parser.add_argument('--past_temp_sprd', type=float, default=None,
                            help='temperature for previous - S-PRD distillation')
        parser.add_argument('--loss_normal_samples', type=str, default='fnc2', choices=['dr', 'fnc2'])
        parser.add_argument('--weight_samix_loss', type=float, default=1.0)

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
        return mean_log_prob_pos  # [2 x bsz,]

    def generate_samix_prototypes_slerp(self, repeated_labels, mixed_indices, corresponding_prototypes, lamb_ratio):
        perm_prototypes = self.points[repeated_labels[mixed_indices]]
        cos_theta = torch.sum(corresponding_prototypes * perm_prototypes, dim=1, keepdim=True)
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        sin_theta = torch.sin(theta)
        slerp_prototypes = torch.empty_like(corresponding_prototypes)
        small_angle_threshold = 1e-6
        small_angle_mask = (theta < small_angle_threshold).squeeze(-1)
        slerp_prototypes[small_angle_mask] = corresponding_prototypes[small_angle_mask]
        non_zero_mask = ~small_angle_mask
        if non_zero_mask.any():
            theta_non_zero = theta[non_zero_mask]
            sin_theta_non_zero = sin_theta[non_zero_mask]
            p1_weight = torch.sin(lamb_ratio * theta_non_zero) / sin_theta_non_zero
            p2_weight = torch.sin((1 - lamb_ratio) * theta_non_zero) / sin_theta_non_zero
            p1_weight = p1_weight.expand(-1, corresponding_prototypes.size(1))
            p2_weight = p2_weight.expand(-1, perm_prototypes.size(1))
            slerp_prototypes[non_zero_mask] = (p1_weight * corresponding_prototypes[non_zero_mask]
                                               + p2_weight * perm_prototypes[non_zero_mask])
        return slerp_prototypes

    def forward(self, features, labels=None, target_labels=None, samix_features=None, lamb_ratio=None,
                mixed_indices=None, **kwargs):
        """
        @param features: [bsz, 2, d]
        @param labels: [bsz,]
        @param target_labels: [n_cls_per_task,]
        @param samix_features: [2 x bsz, d]
        @param lamb_ratio: samixed_views = data * lamb_ratio + (1 - lamb_ratio) * data_perm
        @param mixed_indices: [2 x bsz,]
        @param kwargs:
        @return:
        """
        assert target_labels is not None and len(
            target_labels) > 0, "Target labels should be given as a list of integer"
        assert self.loss_normal_samples in ['dr', 'fnc2']
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

        # For normal features
        if self.loss_normal_samples == 'dr':
            loss_normal_features = torch.sum(anchor_feature * corresponding_prototypes, dim=1)
            loss_normal_features = 0.5 * ((loss_normal_features - 1) ** 2)
        elif self.loss_normal_samples == 'fnc2':
            assert self.focal_gamma is not None
            loss_normal_features = self.compute_fnc2_loss(anchor_feature=anchor_feature, anchor_count=anchor_count,
                                                          labels=labels, batch_size=batch_size, device=device,
                                                          target_labels=target_labels,
                                                          corresponding_prototypes=corresponding_prototypes)
        else:
            raise ValueError(f"args --loss_normal_samples is not valid")

        # Remove old samples as anchors
        curr_class_mask = torch.zeros_like(repeated_labels)
        for tc in target_labels:
            curr_class_mask += (repeated_labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(features)
        loss_normal_features = curr_class_mask * loss_normal_features
        loss_normal_features = torch.mean(loss_normal_features)

        # For samix features
        loss_samix_features = None
        if samix_features is not None and lamb_ratio is not None and mixed_indices is not None:
            mixed_prototypes = self.generate_samix_prototypes_slerp(repeated_labels, mixed_indices,
                                                                    corresponding_prototypes, lamb_ratio)
            # dr - samix samples
            loss_samix_features = torch.sum(samix_features * mixed_prototypes, dim=1)  # [2 x bsz,]
            loss_samix_features = 0.5 * ((loss_samix_features - 1) ** 2)
            loss_samix_features = torch.mean(loss_samix_features)
        return loss_normal_features + self.weight_samix_loss * loss_samix_features
