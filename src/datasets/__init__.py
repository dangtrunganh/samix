from enum import Enum


class Dataset(Enum):
    CIFAR_10 = "cifar10"
    TINY_IMAGENET = "tiny-imagenet"
    CIFAR_100 = "cifar100"
    CUSTOM = "custom"


DATA_INFORMATION = {
    "cifar10": [10, 2, 32],  # n_cls, cls_per_task, image_size
    "tiny-imagenet": [200, 20, 64],
    "cifar100": [100, 20, 32],
    "custom": None
}
