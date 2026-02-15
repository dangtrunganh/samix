import argparse
from src.args.train import utils_training_args
from src.args.dataset import dataset_args
from src.args.continual import continual_args
import os
import math
from src.utils.general_utils import print_x, create_folders, get_feat_dim_projection
from src.backbones.resnet_big import model_dict
from datetime import datetime
from src.plas_losses import LOSSES
from src.datasets import DATA_INFORMATION


def parse_all_training_args(fixed_config) -> argparse.Namespace:
    parser = argparse.ArgumentParser('argument for training')

    # add shared arguments
    dataset_args(parser)
    utils_training_args(parser)
    continual_args(parser)

    temp_args, _ = parser.parse_known_args()
    LOSSES[temp_args.loss].add_specific_args(parser)

    opt = parser.parse_args()

    assert opt.model in model_dict

    opt.save_freq = opt.epochs // 2

    opt.n_cls, opt.cls_per_task, opt.size = DATA_INFORMATION[opt.dataset]
    if opt.num_tasks is not None:
        assert opt.n_cls % opt.num_tasks == 0
        opt.cls_per_task = int(opt.n_cls / opt.num_tasks)

    if opt.dataset == 'custom':
        assert opt.data_folder is not None and opt.mean is not None and opt.std is not None

    opt.dim = get_feat_dim_projection(dataset_name=opt.dataset)

    # Set the path according to the environment
    save_path_config = fixed_config['general']['save_path_config']
    pattern_name = fixed_config['general']['pattern_name']

    root_path = save_path_config['root_save_folder']
    if opt.data_folder is None:
        opt.data_folder = save_path_config['data_folder'].replace("{root_save_folder}", root_path)

    experiments_folder = save_path_config['experiments_folder'].replace("{root_save_folder}", root_path)

    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    job_run_name = opt.job_run_name
    run_exp_folder = save_path_config['run_exp_folder'].replace(
        "{experiments_folder}", experiments_folder).replace(
        "{job_run_name}", job_run_name).replace(
        "{datetime}", current_time)
    if opt.resume_target_task is not None and opt.path_resume_model_folder is not None:
        print_x(f"Resume task: {opt.resume_target_task}, path_resume_model_folder = {opt.path_resume_model_folder}")
        run_exp_folder = opt.path_resume_model_folder

    opt.model_path = '{}/models'.format(run_exp_folder, opt.dataset)
    opt.log_path = '{}/logs'.format(run_exp_folder)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.encoder_folder = 'encoder'
    if opt.cosine:
        opt.encoder_folder = 'encoder_cosine'

    if opt.loss == 'nc_samix' or opt.loss == 'fnc2':
        if opt.dataset == 'tiny-imagenet':
            opt.second_mark = 40
        else:
            opt.second_mark = 80

    # warm-up
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.encoder_folder = '{}_warm'.format(opt.encoder_folder)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.encoder_folder)
    create_folders(opt.save_folder)

    opt.log_folder = os.path.join(opt.log_path, opt.encoder_folder)
    create_folders(opt.log_folder)

    # wandb dir
    if opt.wandb:
        opt.wandb_dir = root_path
        opt.wandb_run_name_train = pattern_name['job_wandb_name_train'] \
            .replace('{job_run_name}', job_run_name)
    return opt
