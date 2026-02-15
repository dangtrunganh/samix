from argparse import ArgumentParser
from src.plas_losses import LOSSES
from src.args.utils import general_args
from src.distillation import LIST_DISTILLATION_LOSSES


def utils_training_args(parser: ArgumentParser):
    """Adds all arguments to a parser.
    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """
    SUPPORTED_METHOD_LOSSES = LOSSES.keys()

    # general args
    general_args(parser)

    # saving models frequency settings
    parser.add_argument('--save_freq', type=int, default=1000,
                        help='save frequency')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=None)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')

    # for customized dataset
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')

    # temperature
    parser.add_argument('--current_temp', type=float, default=0.2,
                        help='temperature for loss function')
    parser.add_argument('--past_temp', type=float, default=0.01,
                        help='temperature for loss function')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='temperature for loss function')

    # memory
    parser.add_argument('--mem_size', type=int, default=200)

    # loss
    parser.add_argument('--loss', type=str, default='nc_samix', choices=SUPPORTED_METHOD_LOSSES)
    parser.add_argument('--distillation', type=str, default='hsd', choices=LIST_DISTILLATION_LOSSES)

    # for resume training
    parser.add_argument('--resume_target_task', type=int, default=None)
    parser.add_argument('--path_resume_model_folder', type=str, default=None)
    parser.add_argument('--path_model_resume_task', type=str, default=None)
    parser.add_argument('--path_replay_indices', type=str, default=None)

    # filter protos from task 0 to curren task for distillation
    parser.add_argument('--filter_distillation_point', action='store_true', default=False)

    # In case which mem = 0, set as 'not-use' with mem > 0
    parser.add_argument('--log_folder_zero_mem', type=str, default='')

    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--eval_epochs', type=int, default=100)
    parser.add_argument('--eval_learning_rate', type=float, default=1.0)
    parser.add_argument('--eval_num_workers', type=int, default=16)

    # hsd loss
    parser.add_argument('--main_mark', type=int, default=20)

    # samix
    parser.add_argument('--samix', action='store_true', default=False)
