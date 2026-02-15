from argparse import ArgumentParser
from src.args.utils import general_args


def evaluate_args(parser: ArgumentParser):
    """Adds all arguments to a parser.
    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """
    general_args(parser)
    parser.add_argument('--target_task', type=int, default=0, help='Use all classes if None else learned tasks so far')

    # saving models frequency settings
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

    # load model path - ckpt & log folder
    parser.add_argument('--ckpt', type=str, default='', help='path to pre-trained model')
    parser.add_argument('--logpt', type=str, default='', help='path to pre-trained model')

    # If using all train dataset for evaluating
    parser.add_argument('--no_eval_mem', action='store_true', default=False)
