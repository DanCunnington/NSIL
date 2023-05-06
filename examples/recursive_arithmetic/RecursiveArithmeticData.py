import copy

from src.NSILDataset import NSILDataset, NSILImageLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from global_config import IMAGE_DATA_DIR
from torch.utils.data import Dataset, DataLoader
from os.path import join

import yaml
import math


def get_data(args, yaml_file, train=False):
    if train:
        if args.meta_abd_data:
            num_examples = 3000
        else:
            num_examples = math.floor((args.pct / 100) * len(yaml_file))
    else:
        num_examples = len(yaml_file)
    data = yaml_file[:num_examples]
    return data


class RecursiveArithmeticDataset(Dataset):
    def __init__(self, args, yaml_file, images, train=False):
        """
        The PyTorch Dataset class for this task
        @param args: cmd line arguments
        @param yaml_file: YAML file containing the data
        @param images: MNIST image dataset
        @param train: whether this is the training set or test set. Used to reduce dataset percentage
        """
        self.args = args
        self.yaml_file = yaml_file
        self.images = images
        self.data = get_data(self.args, self.yaml_file, train)
        self.unique_labels = set([i.y for i in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item.x_idxs, item.y, item.x


class RecursiveArithmeticData(NSILDataset):
    def __init__(self, runner,
                 root_dir,
                 train_file='mysum_full.yaml',
                 test_file='mysum_full_test.yaml'):
        """
        NSILDataset Instance
        @param runner: the NSILRun instance
        @param root_dir: directory containing the datasets
        @param train_file: path to train file
        @param test_file: path to test file
        """
        runner.logger.info('load_data')
        self.train_file = train_file
        self.test_file = test_file
        if 'sum' in test_file:
            self.extra_test_files = [
                'mysum_full_test_10.yaml',
                'mysum_full_test_100.yaml'
            ]
        else:
            self.extra_test_files = [
                'myprod_full_test_10.yaml',
                'myprod_full_test_15.yaml'
            ]

        with open(join(root_dir, self.train_file), 'r') as f:
            self.train_yaml = yaml.load(f, Loader=yaml.Loader)

        with open(join(root_dir, self.test_file), 'r') as f:
            self.test_yaml = yaml.load(f, Loader=yaml.Loader)

        self.extra_test_yamls = {}
        for yf in self.extra_test_files:
            with open(join(root_dir, yf), 'r') as f:
                self.extra_test_yamls[yf] = yaml.load(f, Loader=yaml.Loader)

        self.MNIST_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.MNIST_data = {
            'train': MNIST(root=IMAGE_DATA_DIR, train=True, download=False, transform=self.MNIST_transform),
            'test': MNIST(root=IMAGE_DATA_DIR, train=False, download=False, transform=self.MNIST_transform)
        }
        super().__init__(runner, root_dir)

    def load_nesy_data(self):
        train_data = RecursiveArithmeticDataset(self.args, self.train_yaml, self.MNIST_data['train'], train=True)
        test_data = RecursiveArithmeticDataset(self.args, self.test_yaml, self.MNIST_data['test'])
        train_loader = DataLoader(train_data, batch_size=1, num_workers=self.args.num_workers)
        test_loader = DataLoader(test_data, batch_size=1, num_workers=self.args.num_workers)
        # No validation loader as using Meta_Abd data
        return {
            'train': train_loader,
            'val': None,
            'test': test_loader
        }

    def load_extra_test_nesy_data(self):
        loaders = {}
        for yf in self.extra_test_yamls:
            extra_data = RecursiveArithmeticDataset(self.args, self.extra_test_yamls[yf], self.MNIST_data['test'])
            l = DataLoader(extra_data, batch_size=1, num_workers=self.args.num_workers)
            loaders[yf] = l
        return loaders

    def load_nn_data(self):
        # Remove images from MNIST training set that aren't in nesy data
        nesy_train = get_data(self.args, self.train_yaml, train=True)
        x_idxs = [i.x_idxs for i in nesy_train]
        # Flatten
        nesy_x_idxs = [item for sublist in x_idxs for item in sublist]

        # Extract train_images
        train_images = []
        train_labels = []
        train_idx_map = []
        for idx, item in enumerate(self.MNIST_data['train']):
            if idx in nesy_x_idxs:
                train_images.append(item[0])
                train_labels.append(item[1])
                train_idx_map.append(idx)

        new_train_ds = NSILImageLoader(train_images, train_labels, train_idx_map)

        train_loader = DataLoader(new_train_ds, batch_size=self.args.net_batch_size,
                                  num_workers=self.args.num_workers)
        test_loader = DataLoader(self.MNIST_data['test'], batch_size=self.args.net_batch_size,
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
        data = get_data(self.args, self.train_yaml, train=True)
        data_list = []
        obs_list = []
        for item in data:
            d_list_entry = {}
            for idx, image_idx in enumerate(item.x_idxs):
                i = f'i{idx + 1}'
                d_list_entry[i] = self.MNIST_data['train'][image_idx][0]
            data_list.append(d_list_entry)
            obs_list.append(f':- not result({item.y}).')
        return data_list, obs_list


