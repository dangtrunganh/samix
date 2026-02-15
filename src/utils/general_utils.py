from __future__ import print_function

import os
import inspect
import math
import numpy as np
import torch
import random
import torch.optim as optim
from torch.utils.data import Dataset
from src.datasets.tiny_imagenet_dataset import TinyImagenet
from torchvision import transforms, datasets
from src.datasets import Dataset
import json


def print_x(s):
    caller_frame = inspect.stack()[1]
    caller_file_path = caller_frame.filename
    caller_file_name = os.path.basename(caller_file_path)
    print(f"[{caller_file_name}] {s}")


def get_train_datasets(opt, data_transform):
    if opt.dataset == 'cifar10':
        return datasets.CIFAR10(root=opt.data_folder,
                                transform=data_transform,
                                download=True)
    elif opt.dataset == 'tiny-imagenet':
        return TinyImagenet(root=opt.data_folder,
                            transform=data_transform,
                            download=True)
    elif opt.dataset == 'cifar100':
        return datasets.CIFAR100(root=opt.data_folder,
                                 transform=data_transform,
                                 download=True)
    elif opt.dataset == 'custom':
        return datasets.ImageFolder(root=opt.data_folder,
                                    transform=data_transform)
    else:
        raise ValueError('Dataset not supported: {}'.format(opt.dataset))


def get_val_datasets(opt, val_transform):
    if opt.dataset == 'cifar10':
        return datasets.CIFAR10(root=opt.data_folder,
                                train=False,
                                transform=val_transform)
    elif opt.dataset == 'tiny-imagenet':
        return TinyImagenet(root=opt.data_folder,
                            transform=val_transform,
                            train=False)
    elif opt.dataset == 'cifar100':
        return datasets.CIFAR100(root=opt.data_folder,
                                 train=False,
                                 transform=val_transform)
    elif opt.dataset == 'custom':
        return datasets.ImageFolder(root=opt.data_folder,
                                    transform=val_transform)
    else:
        raise ValueError('Dataset not supported: {}'.format(opt.dataset))


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print_x('==> Saving...' + save_file)
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def load_model(model, optimizer, save_file):
    print_x('==> Loading...' + save_file)
    loaded = torch.load(save_file)
    model.load_state_dict(loaded['model'])
    optimizer.load_state_dict(loaded['optimizer'])
    del loaded
    return model, optimizer


def set_replay_samples(opt, model, prev_indices=None):
    is_training = model.training
    model.eval()
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_dataset = get_train_datasets(opt, val_transform)
    val_targets = val_dataset.targets
    val_targets = np.array(val_targets)

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
    print_x(np.unique(val_targets[selected_observed_indices], return_counts=True))

    model.is_training = is_training
    output = prev_indices + selected_observed_indices
    return output


def save_config_file(opt, save_path):
    dict_opt = vars(opt)
    with open(save_path, 'w') as f:
        json.dump(dict_opt, f, indent=4)
        f.close()


def create_folders(dirs):
    if not os.path.isdir(dirs):
        os.makedirs(dirs)


def get_feat_dim_projection(dataset_name):
    if dataset_name == Dataset.TINY_IMAGENET.value:
        return 256
    return 128


def write_eval_acc(accuracy_single_task_path, best_acc, val_acc,
                   dict_acc_classes, best_task_acc, val_task_acc):
    with open(accuracy_single_task_path, 'w') as f:
        out = 'Best class-il accuracy: {:.2f}, task-il accuracy: {:.2f}\n'.format(best_acc, best_task_acc)
        out += '{:.2f} {:.2f}'.format(val_acc, val_task_acc)
        print_x(out)
        out += '\n'
        for k, v in dict_acc_classes.items():
            print_x(v)
            out += f'class: {k} - acc: {v:.2f}\n'
        f.write(out)


def compute_forgetting(final_dict_acc_tasks, maximum_acc):
    list_subtract = [maximum_acc[i][1] - final_dict_acc_tasks[i] for i in final_dict_acc_tasks.keys()][:-1]
    return sum(list_subtract) / len(list_subtract)
