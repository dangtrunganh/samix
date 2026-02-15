import numpy as np
import torch
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
from src.datasets import DATA_INFORMATION
from src.utils.transforms_utils import get_transforms
from src.utils.general_utils import get_train_datasets, get_val_datasets


def set_loader(opt, replay_indices):
    replay_indices = replay_indices.tolist()
    train_transform, val_transform = get_transforms(opt)
    all_target_classes = list(range(0, (opt.target_task + 1) * opt.cls_per_task))
    _train_dataset = get_train_datasets(opt, train_transform)
    if opt.dataset in DATA_INFORMATION:
        subset_indices = []
        _train_targets = np.array(_train_dataset.targets)
        start_cls = opt.target_task * opt.cls_per_task
        if opt.no_eval_mem:
            start_cls = 0
            replay_indices = []

        for tc in range(start_cls, (opt.target_task + 1) * opt.cls_per_task):
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
        subset_indices += replay_indices
        ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)

        weights = np.array([0.] * len(subset_indices))
        for t, c in zip(ut, uc):
            weights[_train_targets[subset_indices] == t] = 1. / c

        train_dataset = Subset(_train_dataset, subset_indices)
        subset_indices = []
        _val_dataset = get_val_datasets(opt, val_transform)
        for tc in all_target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset = Subset(_val_dataset, subset_indices)
    else:
        raise ValueError('Dataset not supported: {}'.format(opt.dataset))
    train_sampler = WeightedRandomSampler(torch.Tensor(weights), len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, val_loader, uc
