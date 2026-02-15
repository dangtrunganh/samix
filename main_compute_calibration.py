import torch
from src.backbones.resnet_big import SupConResNet, LinearClassifier
import torch.backends.cudnn as cudnn
from src.utils.transforms_utils import get_transforms
from torchvision import datasets
from torch.utils.data import Dataset, Subset
import numpy as np
from src.utils.general_utils import print_x, get_feat_dim_projection
import argparse
from config.config_reader import read_config
from src.datasets import DATA_INFORMATION

device = torch.device("cpu")
if torch.cuda.is_available():
    torch.backends.cuda.max_split_size_mb = 128
    device = torch.device("cuda")
    print_x("CUDA available")
else:
    print_x("CPU available")


def expected_calibration_and_overconfidence_error(predictions, labels, n_bins):
    """
    Compute Expected Calibration Error (ECE) and Overconfidence Error (OE).

    Args:
        predictions (torch.Tensor): Model outputs (logits or probabilities).
        labels (torch.Tensor): Ground truth labels.
        n_bins (int): Number of bins for confidence calibration.

    Returns:
        tuple: (ECE, OE)
    """
    probs = torch.softmax(predictions, dim=1)
    confidences, predicted_classes = torch.max(probs, dim=1)
    avg_confidence = confidences.mean()
    accuracies = (predicted_classes == labels).float()
    acc = accuracies.mean()
    ece = 0.0
    oe = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_accs = torch.zeros(n_bins)
    bin_confs = torch.zeros(n_bins)
    bin_sizes = torch.zeros(n_bins)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if mask.sum().item() > 0:
            bin_sizes[i] = mask.sum().float()
            bin_accs[i] = accuracies[mask].mean()
            bin_confs[i] = confidences[mask].mean()
            bin_weight = mask.float().mean()
            ece += bin_weight * torch.abs(bin_accs[i] - bin_confs[i])
            overconfidence = torch.clamp(bin_confs[i] - bin_accs[i], min=0)
            oe += bin_weight * bin_confs[i] * overconfidence
    return ece.item(), oe.item(), bin_accs.detach().numpy(), bin_confs.detach().numpy(), bin_sizes.detach().numpy(), avg_confidence, acc


def load_classifier(path_classifier):
    classifier = LinearClassifier(name="resnet18", num_classes=opt.n_cls)
    ckpt_classifier = torch.load(path_classifier, map_location='cpu')
    state_dict_classifier = ckpt_classifier['model']
    classifier.load_state_dict(state_dict_classifier)
    return classifier


def load_backbone(path_checkpoint, predictor_hidden_dim):
    feat_dim = get_feat_dim_projection(opt.dataset)
    model = SupConResNet(name="resnet18", feat_dim=feat_dim, head='mlp', predictor_hidden_dim=predictor_hidden_dim)
    ckpt = torch.load(path_checkpoint, map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model = model.cpu()
    cudnn.benchmark = True
    model.load_state_dict(state_dict)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Calibration computation for CL')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny-imagenet'],
                        help='dataset name (default: cifar10)')
    parser.add_argument('--target_task', type=int, default=4,
                        help='target task number (default: 4)')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='number of bins to calculate calibration')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='path to model checkpoint')
    parser.add_argument('--classifier_path', type=str, required=True,
                        help='path to linear classifier checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    fixed_config = read_config()
    save_path_config = fixed_config['general']['save_path_config']
    root_path = save_path_config['root_save_folder']
    data_folder = save_path_config['data_folder'].replace("{root_save_folder}", root_path)
    opt = parse_args()
    opt.n_cls, opt.cls_per_task, opt.size = DATA_INFORMATION[opt.dataset]

    # data
    train_transform, val_transform = get_transforms(opt)
    all_target_classes = list(range(0, (opt.target_task + 1) * opt.cls_per_task))
    subset_indices = []
    _val_dataset = datasets.CIFAR100(root=data_folder, train=False, download=True, transform=val_transform)
    for tc in all_target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset = Subset(_val_dataset, subset_indices)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    data, labels = next(iter(val_loader))

    images = data.float().to(device)
    labels = labels.to(device)

    num_tasks = opt.n_cls // opt.cls_per_task
    backbone = load_backbone(path_checkpoint=opt.checkpoint_path,
                             predictor_hidden_dim=get_feat_dim_projection(opt.dataset))
    linear_classifier = load_classifier(path_classifier=opt.classifier_path)
    if torch.cuda.is_available():
        backbone = backbone.cuda()
        linear_classifier = linear_classifier.cuda()
    features = backbone.encoder(images)

    sum_ece = 0.0
    sum_oe = 0.0
    for task_idx in range(num_tasks):
        lower_bound = task_idx * opt.cls_per_task
        upper_bound = (task_idx + 1) * opt.cls_per_task
        mask = (labels >= lower_bound) & (labels < upper_bound)
        filtered_features = features[mask]
        filtered_labels = labels[mask]
        output_logits = linear_classifier(filtered_features)
        ECE, OE, accuracies, confidences, bin_sizes, avg_confidence, acc = expected_calibration_and_overconfidence_error(
            predictions=output_logits, labels=filtered_labels, n_bins=opt.n_bins)
        sum_ece += ECE
        sum_oe += OE
        print_x(f"Task-{task_idx + 1}: classes {lower_bound} to {upper_bound - 1}")
        print_x(
            f"ECE = {ECE:.4f}, OE = {OE:.8f}, avg_confidence = {avg_confidence:.4f}, acc = {acc:.4f}, \nbin_sizes = {bin_sizes},\naccuracies = {accuracies}, \nconfidences = {confidences}")
        print_x(f"-----------------------")

    print_x(f"Average across tasks: AECE = {(sum_ece / num_tasks):.4f}, AOE = {(sum_oe / num_tasks):.8f}")
