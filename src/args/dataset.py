from argparse import ArgumentParser
from pathlib import Path


def dataset_args(parser: ArgumentParser):
    """Adds dataset-related arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """

    SUPPORTED_DATASETS = [
        "cifar10",
        "cifar100",
        "tiny-imagenet",
        "custom",
    ]
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=SUPPORTED_DATASETS, required=True, help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument("--train_dir", type=Path, default=None, help='train_dir use for custom dataset')
    parser.add_argument("--val_dir", type=Path, default=None, help='val_dir use for custom dataset')
