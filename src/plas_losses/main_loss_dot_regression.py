# -*- coding:  utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
from src.plas_losses.base_prototype_loss import BasePrototypeBasedLoss
from src.plas_losses.base_loss import BaseLoss
from src.utils.general_utils import print_x
import argparse


class DotRegression(BaseLoss, BasePrototypeBasedLoss, nn.Module):
    def set_set_prototypes(self, set_prototypes):
        super().set_set_prototypes(set_prototypes)

    def __init__(self,
                 dim: int = None,
                 n_cls: int = None,
                 **kwargs):
        nn.Module.__init__(self)
        BasePrototypeBasedLoss.__init__(self)
        BaseLoss.__init__(self)
        print_x(f"**kwargs = {kwargs}")
        self.d = dim
        self.k = n_cls
        print_x(self)

    def __str__(self):
        lines = [f"Fixed prototypes: (dim_d={self.d}, n_cls={self.k})"]
        return "[Dot-Regression]" + "\n" + "\n".join(["\t" + line for line in lines])

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser):
        print_x(f"Adding specific args....")
        parser.add_argument('--current_temp_sprd', type=float, default=None,
                            help='temperature for current - S-PRD distillation')
        parser.add_argument('--past_temp_sprd', type=float, default=None,
                            help='temperature for previous - S-PRD distillation')

    def forward(self, features, labels=None, target_labels=None, **kwargs):
        """
        @param features: [bsz, 2, d]
        @param labels: [bsz,]
        @param target_labels: [n_cls_per_task,]
        @param kwargs:
        @param return:
       """
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
        dot_product = torch.sum(anchor_feature * corresponding_prototypes, dim=1)

        curr_class_mask = torch.zeros_like(repeated_labels)
        for tc in target_labels:
            curr_class_mask += (repeated_labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(features)
        dot_product = curr_class_mask * dot_product
        return torch.mean(0.5 * ((dot_product - 1) ** 2))
