'''
    This is experimental code for evaluating a single task
'''

from __future__ import print_function
import sys
import time
import numpy as np
import copy
import torch
import torch.backends.cudnn as cudnn
from src.datasets.evaluate_dataloader import set_loader
from src.utils.general_utils import AverageMeter
from src.utils.general_utils import adjust_learning_rate, warmup_learning_rate
from src.utils.general_utils import set_optimizer
from src.utils.metrics import TaskAccuracy
from src.backbones.resnet_big import SupConResNet, LinearClassifier
from config.config_reader import read_config
from src.utils.general_utils import print_x, save_config_file, write_eval_acc, \
    get_feat_dim_projection
from src.args.setup_evaluating import parse_all_evaluating_args, parse_general_evaluating_args
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_model(opt):
    feat_dim = get_feat_dim_projection(opt.dataset)
    if hasattr(opt, "predictor_hidden_dim") and opt.predictor_hidden_dim:
        model = SupConResNet(name=opt.model, feat_dim=feat_dim, head='mlp', predictor_hidden_dim=opt.predictor_hidden_dim)
    else:
        model = SupConResNet(name=opt.model, feat_dim=feat_dim, head='mlp')
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict)
    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    acc = 0.0
    cnt = 0.0
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc += (output.argmax(1) == labels).float().sum().item()
        cnt += bsz

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if opt.wandb:
            wandb.log({"linear train/loss": losses.val,
                       "linear train/acc_top1": acc / cnt * 100.})
        if (idx + 1) % opt.print_freq == 0:
            print_x('Train: Epoch:[{0}] batch [{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Acc@1 {top1:.3f}'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=acc / cnt * 100.))
            sys.stdout.flush()

    return losses.avg, acc / cnt * 100.


def validate(val_loader, model, classifier, criterion, epoch, opt):
    """validation each epoch"""
    model.eval()
    classifier.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    corr = [0.] * (opt.target_task + 1) * opt.cls_per_task
    cnt = [0.] * (opt.target_task + 1) * opt.cls_per_task
    correct_task = 0.0

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().to(device)
            labels = labels.to(device)
            batch_size = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            cls_list = np.unique(labels.cpu())
            correct_all = (output.argmax(1) == labels)

            for tc in cls_list:
                mask = labels == tc
                # task-il acc
                check_tc = output[mask,
                           (tc // opt.cls_per_task) * opt.cls_per_task: ((
                                                                                 tc // opt.cls_per_task) + 1) * opt.cls_per_task]
                correct_task += (check_tc.argmax(1) == (tc % opt.cls_per_task)).float().sum()

                # class-il acc
                corr[tc] += correct_all[mask].float().sum().item()
                cnt[tc] += mask.float().sum().item()

            if idx % opt.print_freq == 0:
                print_x('Test: Epoch: [{0}] batch [{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1:.3f} {task_il:.3f}'.format(
                    epoch, idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=np.sum(corr) / np.sum(cnt) * 100., task_il=correct_task / np.sum(cnt) * 100.))
    if opt.wandb:
        wandb.log({"val/loss": losses.val,
                   "val/acc_top1": np.sum(corr) / np.sum(cnt) * 100.,
                   "val/acc_task_il": correct_task / np.sum(cnt) * 100.})
    print_x(
        ' * Epoch {0} - Acc@1={top1:.3f} task_il={task_il:.3f}'.format(epoch,
                                                                       top1=np.sum(corr) / np.sum(cnt) * 100.,
                                                                       task_il=correct_task / np.sum(cnt) * 100.))
    return losses.avg, corr, cnt, correct_task / np.sum(cnt) * 100.


def evaluate_single_task(opt):
    overall_acc_task = TaskAccuracy(opt.target_task, opt.epochs)
    best_model = None

    print_x(f"Args = {opt}")
    save_config_file(opt, opt.eval_config_path)
    if opt.target_task is not None:
        if opt.target_task == 0:
            replay_indices = np.array([])
        else:
            replay_indices = np.load(opt.logpt)
        print_x(f"len(replay_indices) = {len(replay_indices)}")
    # build data loader
    train_loader, val_loader, cls_num_list = set_loader(opt, replay_indices)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        print_x(f"Start eval epoch: {epoch}/{opt.epochs}")
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        print_x("-------------Train classifier------")
        print_x('Train: [EPOCH][idx in batch/number of batch]\t'
                'BT batch_time.val (batch_time.avg)\t'
                'DT data_time.val (data_time.avg)\t'
                'loss loss.val (loss.avg)\t'
                'Acc@1 top1_acc')
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion, optimizer, epoch, opt)
        print_x('Train epoch: {}, total time {:.2f}, loss: {:.2f}, accuracy: {:.2f}, lr: {:.3f}'.format(
            epoch, time.time() - time1, loss, acc, optimizer.param_groups[0]['lr']))

        # eval for one epoch
        print_x("-------------Test classifier------")
        print_x('Test: [batch_idx/total_batches]\t'
                'Time batch_time.val (batch_time.avg)\t'
                'Loss loss.val (loss.avg)\t'
                'Acc@1 top1_acc task_il')
        loss, val_corr, val_cnt, task_acc = validate(val_loader, model, classifier, criterion, epoch, opt)
        val_acc = np.sum(val_corr) / np.sum(val_cnt) * 100.
        if val_acc > overall_acc_task.best_acc[1]:
            overall_acc_task.best_acc = (epoch, val_acc)
            overall_acc_task.best_task_acc = (epoch, task_acc)
            best_model = copy.deepcopy(classifier)
            val_acc_stats = {}
            for cls, (cr, c) in enumerate(zip(val_corr, val_cnt)):
                if c > 0:
                    val_acc_stats[str(cls)] = cr / c * 100.
            overall_acc_task.dict_acc_classes = val_acc_stats

        # wandb log - per epoch
        if opt.wandb:
            wandb.log({f"overall/train_acc": acc,
                       f"overall/val_acc": val_acc,
                       f"overall/task_acc": task_acc,
                       f"overall/epoch": epoch})

    # end of task
    overall_acc_task.last_acc = val_acc
    overall_acc_task.task_acc = task_acc
    # wandb log
    if opt.wandb:
        wandb.run.summary[f"Task-{opt.target_task}: best_accuracy"] = overall_acc_task.best_acc[1]
        wandb.run.summary[f"Task-{opt.target_task}: best_task_il_accuracy"] = overall_acc_task.best_task_acc[1]
        wandb.run.summary[f"Task-{opt.target_task}: best_accuracy at epoch"] = overall_acc_task.best_acc[0]
        wandb.run.summary[f"Task-{opt.target_task}: last_accuracy"] = val_acc
        wandb.run.summary[f"Task-{opt.target_task}: last_task_il_accuracy"] = task_acc

    # report
    print_x(f"===> Report acc result into path: {opt.accuracy_single_task_path}")
    write_eval_acc(opt.accuracy_single_task_path, overall_acc_task.best_acc[1], val_acc,
                   overall_acc_task.dict_acc_classes, overall_acc_task.best_task_acc[1], task_acc)

    # save model
    print_x('==> Saving model into path:...' + opt.eval_model_path)
    torch.save({
        'opt': opt,
        'model': best_model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, opt.eval_model_path)
    return overall_acc_task


if __name__ == '__main__':
    fixed_config = read_config()
    opt = parse_general_evaluating_args()
    opt = parse_all_evaluating_args(fixed_config, opt)
    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if opt.wandb:
        wandb.init(project=opt.wandb_project_name, name=opt.wandb_run_name_eval, entity=opt.wandb_entity,
                   dir=opt.wandb_dir, config=vars(opt))

    overall_acc_task = evaluate_single_task(opt=opt)
