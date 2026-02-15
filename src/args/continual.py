from argparse import ArgumentParser


def continual_args(parser: ArgumentParser):
    """Adds continual learning arguments to a parser.

    Args:
        parser (ArgumentParser): parser to add dataset args to.
    """
    parser.add_argument('--target_task', type=int, default=0)
    parser.add_argument('--end_task', type=int, default=None)
    parser.add_argument('--cls_per_task', type=int, default=2, help='Use for class incremental learning')
    SPLIT_STRATEGIES = ["class", "data", "domain"]
    parser.add_argument("--split_strategy", choices=SPLIT_STRATEGIES, type=str, default='class')
    # distillation args
    parser.add_argument("--distiller", type=str, default=None)
    parser.add_argument('--distill_power', type=float, default=1.0)
    parser.add_argument('--predictor_hidden_dim', type=int, default=None)
