from __future__ import print_function
from torchvision import transforms


def get_transforms(opt):
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset == 'tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size // 20 * 2 + 1, sigma=(0.1, 2.0))],
                               p=0.5 if opt.size > 32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform
