import numpy as np
import torch
from torch.utils.data import Subset
from src.utils.general_utils import TwoCropTransform, get_train_datasets
from src.utils.transforms_utils import get_transforms


def set_loader_with_replay(opt, replay_indices):
    train_transform, _ = get_transforms(opt)
    all_target_classes = list(range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task))
    subset_indices = []
    _train_dataset = get_train_datasets(opt, TwoCropTransform(train_transform))
    for tc in all_target_classes:
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
    subset_indices += replay_indices
    train_dataset = Subset(_train_dataset, subset_indices)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    return train_loader, subset_indices
