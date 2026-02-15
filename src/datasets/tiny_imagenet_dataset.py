import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """

    def __init__(self, root: str, train: bool = True, transform: transforms = None,
                 target_transform: transforms = None, download: bool = False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.extracted_folder = os.path.join(root, 'tiny-imagenet')
        if not os.path.isdir(self.extracted_folder):
            os.mkdir(self.extracted_folder)
        if download:
            print(f"[TINY IMAGENET] Download Tiny Imagenet dataset...")
            if os.path.isdir(self.extracted_folder) and len(os.listdir(self.extracted_folder)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from google_drive_downloader import GoogleDriveDownloader as gdd
                print('Downloading dataset')
                dest_path = os.path.join(self.extracted_folder, 'tiny-imagenet-processed.zip')
                gdd.download_file_from_google_drive(
                    file_id='id_file_name',  # replace with actual file ID - Due to anonymity requirements, please set this manually using your own Google Drive file ID
                    dest_path=dest_path,
                    unzip=True)
                # unzip again
                print('Finish downloading dataset')
        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(self.extracted_folder,
                                                  'processed/x_%s_%02d.npy' % (
                                                      'train' if self.train else 'val', num + 1))))
        self.data = np.concatenate(np.array(self.data))
        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(self.extracted_folder,
                                                     'processed/y_%s_%02d.npy' % (
                                                         'train' if self.train else 'val', num + 1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]
        return img, target
