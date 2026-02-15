from __future__ import print_function

import os
import copy
import time
import wandb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from src.utils.general_utils import (AverageMeter, adjust_learning_rate, warmup_learning_rate, print_x, set_optimizer,
                                     save_model, load_model, set_replay_samples,
                                     save_config_file)
from src.backbones.resnet_big import SupConResNet
from src.datasets.pretrain_dataloader import set_loader_with_replay
from src.plas_losses import LOSSES, LIST_LOSSES_PROTOTYPES
from config.config_reader import read_config
from src.args.setup_training import parse_all_training_args
from src.distillation.stab_loss import compute_distillation_loss
from src.utils.mix_views_generation import generate_samix
from src.utils.prototypes import generate

device = torch.device("cpu")
if torch.cuda.is_available():
    torch.backends.cuda.max_split_size_mb = 128
    device = torch.device("cuda")
    print_x("CUDA available")
else:
    print_x("CPU available")


def get_prototypes(criterion, opt):
    n_proto_per_cls = opt.n_proto_per_cls if hasattr(opt, "n_proto_per_cls") else 1
    current_prototypes = criterion.new_prototypes if hasattr(criterion, "new_prototypes") else criterion.points
    indices_prototypes_target_classes = list(
        range(0, (opt.target_task + 1) * opt.cls_per_task * n_proto_per_cls))
    return current_prototypes[indices_prototypes_target_classes]


def train(train_loader, model, frozen_model, criterion, optimizer, epoch, opt):
    """one epoch training
    """
    model.train()
    # Time
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    distill = AverageMeter()
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        # images: [2 x bsz, C, H, W]
        data_time.update(time.time() - end)
        images = torch.cat([images[0], images[1]], dim=0)
        _views, lamb_ratio, mixed_indices = None, None, None
        if hasattr(opt, "samix") and opt.samix:
            _views, lamb_ratio, mixed_indices = generate_samix(data=images)
            if torch.cuda.is_available():
                _views = _views.cuda(non_blocking=True)
                mixed_indices = mixed_indices.cuda(non_blocking=True)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        # warm-up lr
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        distill_predictor_output = None
        if (opt.target_task > 0 and opt.distillation != "no-distillation" and hasattr(opt,
                                                                                      "predictor_hidden_dim") and opt.predictor_hidden_dim and opt.predictor_hidden_dim != 0):
            features, _encoded, distill_predictor_output = model(images, return_feat=True, distill=True)
        else:
            features, _encoded = model(images, return_feat=True)
        row_original_size = features.size(0)
        # samix
        samix_features = None  # [2 x bsz, d]
        samix_distill_predictor_output = None
        if _views is not None:
            if opt.target_task > 0 and opt.distillation != "no-distillation" and hasattr(opt,
                                                                                         "predictor_hidden_dim") and opt.predictor_hidden_dim and opt.predictor_hidden_dim != 0:
                samix_features, samix_encoded, samix_distill_predictor_output = model(_views, return_feat=True,
                                                                                      distill=True)
            else:
                samix_features, samix_encoded = model(_views, return_feat=True)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [bsz, 2, d]
        target_labels = list(range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task))
        loss = criterion(features=features, labels=labels, target_labels=target_labels,
                         samix_features=samix_features, lamb_ratio=lamb_ratio, mixed_indices=mixed_indices)
        if opt.wandb:
            wandb.log({"general_training/plasticity_loss": loss})
        if opt.target_task > 0 and opt.distillation != "no-distillation":
            if distill_predictor_output is not None:
                features1_prev_task = distill_predictor_output
            else:
                features1_prev_task = features.transpose(0, 1).reshape(row_original_size, -1)
            alpha_balance_distillation = None  # increase
            if opt.distillation == 'hsd':
                if hasattr(opt, "main_mark") and hasattr(opt, "second_mark"):
                    if opt.main_mark is not None and opt.second_mark is not None:
                        assert 0 < opt.main_mark < opt.epochs - 1 and opt.main_mark < opt.second_mark <= opt.epochs
                        if epoch < opt.main_mark:
                            alpha_balance_distillation = 0
                        elif opt.main_mark <= epoch < opt.second_mark:
                            alpha_balance_distillation = (epoch - opt.main_mark) / opt.epochs
                        else:
                            alpha_balance_distillation = (opt.second_mark - opt.main_mark) / opt.epochs
                    else:
                        alpha_balance_distillation = (epoch - 1) / opt.epochs
                else:
                    alpha_balance_distillation = (epoch - 1) / opt.epochs
                assert alpha_balance_distillation >= 0
            if _views is not None:
                if samix_distill_predictor_output is not None:
                    features1_prev_task = torch.cat((features1_prev_task, samix_distill_predictor_output), dim=0)
                else:
                    features1_prev_task = torch.cat((features1_prev_task, samix_features), dim=0)
                images = torch.cat((images, _views), dim=0)
            if opt.filter_distillation_point:
                _distill = compute_distillation_loss(opt, features1_prev_task, frozen_model, images,
                                                     set_prototypes=get_prototypes(criterion, opt),
                                                     alpha_balance_distillation=alpha_balance_distillation)
            else:  # no filter proto
                _distill = compute_distillation_loss(opt, features1_prev_task, frozen_model, images,
                                                     set_prototypes=None)
            loss += opt.distill_power * _distill
            distill.update(_distill.item(), bsz)
        losses.update(loss.item(), bsz)
        if opt.wandb:
            wandb.log({"general_training/overall_loss": losses.val})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % opt.print_freq == 0 or idx + 1 == len(train_loader):
            print_x('Train: Epoch: [{0}] batch [{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f} {distill.avg:.3f})\t'
                    'lr {lr: .3f}'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, distill=distill, lr=optimizer.param_groups[0]['lr']))
    return losses.avg, frozen_model, distill.avg


def set_model(opt, set_prototypes=None):
    if hasattr(opt, "predictor_hidden_dim") and opt.predictor_hidden_dim:
        model = SupConResNet(name=opt.model, feat_dim=opt.dim,
                             head='mlp',
                             predictor_hidden_dim=opt.predictor_hidden_dim)
    else:
        model = SupConResNet(name=opt.model, feat_dim=opt.dim,
                             head='mlp')
    LossClass = LOSSES[opt.loss]
    criterion = LossClass(**opt.__dict__)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    if set_prototypes is not None and opt.loss in LIST_LOSSES_PROTOTYPES:
        criterion.set_set_prototypes(set_prototypes)
    return model, criterion


def main():
    fixed_config = read_config()
    pattern_name = fixed_config['general']['pattern_name']
    pattern_model_task_name = pattern_name['model_task_name']
    pattern_replay_indices_task_name = pattern_name['replay_indices_task_name']
    pattern_subset_all_indices_task_name = pattern_name['subset_all_indices_task_name']
    pattern_set_prototypes_name = pattern_name['set_prototypes_name']
    config_json_name_train_name = pattern_name['config_json_name_train']
    opt = parse_all_training_args(fixed_config)
    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    print_x(opt)
    opt.path_save_prototypes = None
    set_prototypes = None
    if opt.loss in LIST_LOSSES_PROTOTYPES:
        path_save_prototypes = os.path.join(opt.log_folder, pattern_set_prototypes_name
                                            .format(d=opt.dim, k=opt.n_cls, seed=seed))
        opt.path_save_prototypes = path_save_prototypes
        set_prototypes = generate(d=opt.dim, k=opt.n_cls, path_save_prototype=path_save_prototypes, seed=seed)
        set_prototypes = torch.from_numpy(set_prototypes)
    if opt.wandb:
        wandb.init(project=opt.wandb_project_name, name=opt.wandb_run_name_train, entity=opt.wandb_entity,
                   dir=opt.wandb_dir, config=vars(opt))

    # model, optimizer
    model, criterion = set_model(opt, set_prototypes)
    frozen_model, _ = set_model(opt)
    frozen_model.eval()
    optimizer = set_optimizer(opt, model)

    # resume running from a specific checkpoint
    replay_indices = None
    if opt.resume_target_task is not None:
        print_x(f"Resume running after task {opt.resume_target_task}...")
        assert (opt.path_resume_model_folder is not None) or (
                opt.path_model_resume_task is not None and opt.path_replay_indices is not None)
        if opt.path_resume_model_folder is not None:
            print_x(f"Resume training, reuse old log folder")
            load_file = os.path.join(opt.save_folder, pattern_model_task_name.format(policy=opt.replay_policy,
                                                                                     target_task=opt.resume_target_task))
            model, optimizer = load_model(model, optimizer, load_file)
            path_resume_replay = os.path.join(opt.log_folder,
                                              pattern_replay_indices_task_name.format(policy=opt.replay_policy,
                                                                                      target_task=opt.resume_target_task))
            replay_indices = np.load(path_resume_replay).tolist()
        elif opt.path_model_resume_task is not None and opt.path_replay_indices is not None:
            print_x(
                f"Resume training, create new log folder")
            model, optimizer = load_model(model, optimizer, opt.path_model_resume_task)
            replay_indices = np.load(opt.path_replay_indices).tolist()

    # separate tasks
    original_epochs = opt.epochs
    if opt.end_task is not None:  # if opt.end_task < T
        if opt.resume_target_task is not None:
            assert opt.end_task > opt.resume_target_task
        opt.end_task = min(opt.end_task + 1, opt.n_cls // opt.cls_per_task)
    else:  # end_task = T
        opt.end_task = opt.n_cls // opt.cls_per_task

    # Train each task
    for target_task in range(0 if opt.resume_target_task is None else opt.resume_target_task + 1, opt.end_task):
        print_x('>>> Start Training current task {}'.format(target_task))
        opt.target_task = target_task
        frozen_model = copy.deepcopy(model)

        path_config_json = os.path.join(
            opt.save_folder, config_json_name_train_name.replace("{target_task}", str(target_task)))
        print_x(f"Saving args task: {target_task} into path: {path_config_json}")
        save_config_file(opt, path_config_json)

        replay_indices = set_replay_samples(opt, model, prev_indices=replay_indices)
        np.save(os.path.join(opt.log_folder, pattern_replay_indices_task_name
                             .format(policy=opt.replay_policy, target_task=target_task)), np.array(replay_indices))
        train_loader, subset_indices = set_loader_with_replay(opt, replay_indices)

        np.save(os.path.join(opt.log_folder, pattern_subset_all_indices_task_name
                             .format(policy=opt.replay_policy, target_task=target_task)), np.array(subset_indices))

        if target_task == 0 and opt.start_epoch is not None:
            opt.epochs = opt.start_epoch
        else:
            opt.epochs = original_epochs

        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)
            time1 = time.time()
            loss, frozen_model, distill = train(train_loader, model, frozen_model, criterion, optimizer, epoch, opt)
            time2 = time.time()
            print_x('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            if opt.wandb:
                wandb.log({f"task_{target_task}/loss": loss,
                           f"task_{target_task}/distill_loss": distill,
                           f"task_{target_task}/learning_rate": optimizer.param_groups[0]['lr'],
                           f"task_{target_task}/epoch": epoch})
        save_file = os.path.join(
            opt.save_folder,
            pattern_model_task_name.format(policy=opt.replay_policy, target_task=target_task))
        save_model(model, optimizer, opt, opt.epochs, save_file)
    print_x(f"Finish training...")


if __name__ == '__main__':
    main()
