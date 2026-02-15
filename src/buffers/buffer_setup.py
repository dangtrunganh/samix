from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
from src.datasets.tiny_imagenet_dataset import TinyImagenet
import random
import math
import torch


def set_replay_samples(opt, model, prev_indices=None):
    is_training = model.training
    model.eval()

    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if opt.dataset == 'cifar10':
        subset_indices = []
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       transform=val_transform,
                                       download=True)
        val_targets = np.array(val_dataset.targets)
    elif opt.dataset == 'cifar100':
        subset_indices = []
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        transform=val_transform,
                                        download=True)
        val_targets = np.array(val_dataset.targets)

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        val_dataset = TinyImagenet(root=opt.data_folder,
                                   transform=val_transform,
                                   download=True)
        val_targets = val_dataset.targets

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    if prev_indices is None:
        prev_indices = []
        observed_classes = list(range(0, opt.target_task * opt.cls_per_task))
    else:
        shrink_size = ((opt.target_task - 1) * opt.mem_size / opt.target_task)
        if len(prev_indices) > 0:
            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices = []
            for c in unique_cls:
                mask = val_targets[_prev_indices] == c
                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))
                if random.random() < p:
                    size_for_c = math.ceil(size_for_c)
                else:
                    size_for_c = math.floor(size_for_c)
                prev_indices += torch.tensor(_prev_indices)[mask][torch.randperm(mask.sum())[:size_for_c]].tolist()
            print(np.unique(val_targets[prev_indices], return_counts=True))
        observed_classes = list(
            range(max(opt.target_task - 1, 0) * opt.cls_per_task, (opt.target_task) * opt.cls_per_task))

    if len(observed_classes) == 0:
        return prev_indices
    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()
    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)

    selected_observed_indices = []
    for c_idx, c in enumerate(val_unique_cls):
        size_for_c_float = (
                (opt.mem_size - len(prev_indices) - len(selected_observed_indices)) / (len(val_unique_cls) - c_idx))
        p = size_for_c_float - ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) // (
                len(val_unique_cls) - c_idx))
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)
        mask = val_targets[observed_indices] == c
        selected_observed_indices += torch.tensor(observed_indices)[mask][
            torch.randperm(mask.sum())[:size_for_c]].tolist()
    print(np.unique(val_targets[selected_observed_indices], return_counts=True))
    model.is_training = is_training
    return prev_indices + selected_observed_indices
