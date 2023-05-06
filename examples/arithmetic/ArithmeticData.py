import copy

from src.NSILDataset import NSILDataset, NSILImageLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from global_config import IMAGE_DATA_DIR
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd
import math


def get_data(args, csv_file, train=False):
    if train:
        num_examples = math.floor((args.pct / 100) * len(csv_file))
    else:
        num_examples = len(csv_file)
    data = csv_file[:num_examples]
    return data.values


class ArithmeticDataset(Dataset):
    def __init__(self, args, csv_file, images, train=False):
        """
        The PyTorch Dataset class for this task
        @param args: cmd line arguments
        @param csv_file: csv file containing the data
        @param images: MNIST image dataset
        @param train: whether this is the training set or test set. Used to reduce dataset percentage
        """
        self.args = args
        self.csv_file = csv_file
        self.images = images
        self.data = get_data(self.args, self.csv_file, train)
        self.unique_labels = sorted(list(self.csv_file['label'].unique()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item[-1]
        x_idxs = list(item[:-1])
        x = [self.images[i][1] for i in x_idxs]
        return x_idxs, label, x


class ArithmeticData(NSILDataset):
    def __init__(self, runner,
                 root_dir,
                 task='sum',
                 images='mnist'):
        """
        NSILDataset Instance
        @param runner: the NSILRun instance
        @param root_dir: directory containing the datasets
        @param task: sum or E9P
        @param images mnist or cifar_10
        """
        runner.logger.info('load_data')
        self.task = task
        self.image_type = images
        self.csv_files = {}

        for f_type in ['train', 'validation', 'test']:
            if self.image_type == 'mnist':
                file_name = join(root_dir, task, f'{f_type}.csv')
            else:
                file_name = join(root_dir, task, 'cifar', f'{f_type}.csv')
            self.csv_files[f_type] = pd.read_csv(file_name)

        if self.image_type == 'cifar_10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image_dset = CIFAR10
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            image_dset = MNIST

        self.image_data = {
            'train': image_dset(root=IMAGE_DATA_DIR, train=True, download=True, transform=transform),
            'test': image_dset(root=IMAGE_DATA_DIR, train=False, download=True, transform=transform)
        }
        super().__init__(runner, root_dir)

    def load_nesy_data(self):
        train_data = ArithmeticDataset(self.args, self.csv_files['train'], self.image_data['train'], train=True)
        val_data = ArithmeticDataset(self.args, self.csv_files['validation'], self.image_data['train'])
        test_data = ArithmeticDataset(self.args, self.csv_files['test'], self.image_data['test'])
        train_loader = DataLoader(train_data, batch_size=1, num_workers=self.args.num_workers)
        val_loader = DataLoader(val_data, batch_size=1, num_workers=self.args.num_workers)
        test_loader = DataLoader(test_data, batch_size=1, num_workers=self.args.num_workers)
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    def load_nn_data(self):
        # Remove images from MNIST training set that aren't in nesy data
        nesy_train = get_data(self.args, self.csv_files['train'], train=True)
        x_idxs = [list(i[:-1]) for i in nesy_train]
        # Flatten
        nesy_x_idxs = [item for sublist in x_idxs for item in sublist]

        # Extract train_images
        train_images = []
        train_labels = []
        train_idx_map = []
        for idx, item in enumerate(self.image_data['train']):
            if idx in nesy_x_idxs:
                train_images.append(item[0])
                train_labels.append(item[1])
                train_idx_map.append(idx)

        new_train_ds = NSILImageLoader(train_images, train_labels, train_idx_map)

        train_loader = DataLoader(new_train_ds, batch_size=self.args.net_batch_size,
                                  num_workers=self.args.num_workers)
        test_loader = DataLoader(self.image_data['test'], batch_size=self.args.net_batch_size,
                                 num_workers=self.args.num_workers)
        nn_data = {
            'digit': {
                'train': train_loader,
                'test': test_loader
            }
        }
        return nn_data

    def convert_to_NeurASP(self):
        """
        Convert data into data_list and obs_list for NeurASP training
        """
        data = get_data(self.args, self.csv_files['train'], train=True)
        data_list = []
        obs_list = []
        for item in data:
            d_list_entry = {}
            for idx, image_idx in enumerate(list(item[:-1])):
                i = f'i{idx + 1}'
                d_list_entry[i] = self.image_data['train'][image_idx][0]
            data_list.append(d_list_entry)
            obs_list.append(f':- not result({item[-1]}).')
        return data_list, obs_list


