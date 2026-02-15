from argparse import ArgumentParser, Namespace
from src.args.evaluate import evaluate_args
from src.args.dataset import dataset_args
from src.utils.general_utils import print_x, create_folders
import os
import math
from src.datasets import DATA_INFORMATION
from datetime import datetime


def parse_general_evaluating_args():
    parser = ArgumentParser('argument for evaluating')
    dataset_args(parser)
    evaluate_args(parser)
    opt = parser.parse_args()
    return opt


def parse_all_evaluating_args(fixed_config, opt) -> Namespace:
    opt.data_folder = None

    # config
    save_path_config = fixed_config['general']['save_path_config']
    pattern_name = fixed_config['general']['pattern_name']
    root_path = save_path_config['root_save_folder']
    if opt.data_folder is None:
        opt.data_folder = save_path_config['data_folder'].replace("{root_save_folder}", root_path)

    # eval folder
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    eval_folder_name = f"eval_linear_{current_time}"

    opt.eval_folder_path = os.path.join(opt.ckpt, '..', eval_folder_name)
    create_folders(opt.eval_folder_path)

    opt.eval_model_path = os.path.join(opt.eval_folder_path,
                                       pattern_name['eval_model_name'].replace(
                                           "{eval_folder_name}", eval_folder_name).replace(
                                           "{target_task}", str(opt.target_task)))

    opt.eval_config_path = os.path.join(opt.eval_folder_path,
                                        pattern_name['config_json_name_eval'].replace(
                                            "{target_task}", str(opt.target_task)))

    opt.accuracy_single_task_path = os.path.join(opt.eval_folder_path,
                                                 pattern_name['accuracy_single_task'].replace(
                                                     "{target_task}", str(opt.target_task)))

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.n_cls, opt.cls_per_task, opt.size = DATA_INFORMATION[opt.dataset]
    if opt.num_tasks is not None:
        print_x(f"opt.num_task is not None => process: opt.num_tasks = {opt.num_tasks}")
        assert opt.n_cls % opt.num_tasks == 0
        opt.cls_per_task = int(opt.n_cls / opt.num_tasks)

    opt.predictor_hidden_dim = 256 if opt.dataset == 'tiny-imagenet' else 128

    opt.origin_ckpt = opt.ckpt

    opt.ckpt = os.path.join(opt.ckpt, f'last_{opt.replay_policy}_{opt.target_task}.pth')
    opt.logpt = os.path.join(opt.logpt, f'replay_indices_{opt.replay_policy}_{opt.target_task}.npy')

    # wandb dir
    if opt.wandb:
        opt.wandb_dir = root_path
        opt.wandb_run_name_eval = pattern_name['job_wandb_name_eval'] \
            .replace('{job_run_name}', opt.job_run_name) \
            .replace('{target_task}', str(opt.target_task))
    return opt
