from argparse import ArgumentParser
from src.backbones.resnet_big import model_dict


def general_args(parser: ArgumentParser):
    # training settings
    SUPPORTED_BACKBONES = model_dict.keys()
    parser.add_argument('--seed', type=int, default=5, help='seed number')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    # model backbone
    parser.add_argument('--model', type=str, default='resnet18', choices=SUPPORTED_BACKBONES)
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    # other settings
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # wandb arguments
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project_name', type=str, required=False)
    parser.add_argument('--wandb_entity', type=str, required=False)
    parser.add_argument('--job_run_name', type=str, required=True)
    parser.add_argument('--replay_policy', type=str,
                        default='random', help='Use to find the model name to eval', required=False)
    parser.add_argument('--num_tasks', type=int, default=None, help='Numer of tasks to run')
