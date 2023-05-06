from src.NSILDataset import NSILDataset, NSILImageLoader
from torchvision import transforms
from global_config import IMAGE_DATA_DIR
from torch.utils.data import Dataset, DataLoader
from examples.hitting_sets.templates import TemplateManager
from os.path import join
from skimage import io
import pandas as pd
import math
import torch


def get_data(args, csv_file, train=False):
    if train:
        num_examples = math.floor((args.pct / 100) * len(csv_file))
    else:
        num_examples = len(csv_file)
    data = csv_file[:num_examples]
    return data.values


class Images(Dataset):
    def __init__(self, root_dir, transform=None, gray=True):
        """
        Dataset for both fashionmnist and normal mnist images
        Args:
            root_dir (string): Directory with all the images and labels.csv file
            transform (callable, optional): Optional transform to be applied on a sample.
            gray (bool): whether images are greyscale
        """
        self.root_dir = root_dir
        self.images = pd.read_csv(join(root_dir, 'labels.csv'))
        self.transform = transform
        self.gray = gray

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx >= len(self.images):
            raise IndexError
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = join(self.root_dir, f'{idx}.jpg')
        image = io.imread(img_name, as_gray=self.gray)
        label = self.images.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image.float(), label-1


class HittingSetsDataset(Dataset):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item[-2]
        template = item[-1]
        x_idxs = list(item[:-2])
        x = [self.images[i][1] for i in x_idxs]
        return x_idxs, label, x, template


class HittingSetsData(NSILDataset):
    def __init__(self, runner,
                 root_dir,
                 templates):
        """
        NSILDataset Instance
        @param runner: the NSILRun instance
        @param root_dir: directory containing the datasets
        @param templates: possible set templates
        @param task: hs or chs
        @param image_type: mnist or fashion_mnist
        """
        runner.logger.info('load_data')
        self.task = runner.args.task_type
        self.image_type = runner.args.image_type
        self.csv_files = {}

        for f_type in ['train', 'validation', 'test']:
            file_name = join(root_dir, self.task, f'{self.image_type}_{f_type}.csv')
            self.csv_files[f_type] = pd.read_csv(file_name)

        if self.image_type == 'cifar_10':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            gray = False
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            gray = True
        im_dir = join(IMAGE_DATA_DIR, f'{self.image_type}_1_to_5')
        self.images = {
            'train': Images(root_dir=join(im_dir, 'train'), transform=self.transform, gray=gray),
            'test': Images(root_dir=join(im_dir, 'test'), transform=self.transform, gray=gray)
        }
        self.template_manager = TemplateManager(templates, self.csv_files['train'])
        super().__init__(runner, root_dir)

    def load_nesy_data(self):
        train_data = HittingSetsDataset(self.args, self.csv_files['train'], self.images['train'], train=True)
        val_data = HittingSetsDataset(self.args, self.csv_files['validation'], self.images['train'])
        test_data = HittingSetsDataset(self.args, self.csv_files['test'], self.images['test'])
        train_loader = DataLoader(train_data, batch_size=1, num_workers=self.args.num_workers)
        val_loader = DataLoader(val_data, batch_size=1, num_workers=self.args.num_workers)
        test_loader = DataLoader(test_data, batch_size=1, num_workers=self.args.num_workers)
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    def load_nn_data(self):
        # Remove images from image training set that aren't in nesy data
        nesy_train = get_data(self.args, self.csv_files['train'], train=True)
        x_idxs = [list(i[:-2]) for i in nesy_train]
        nesy_x_idxs = [item for sublist in x_idxs for item in sublist]

        # Extract train_images
        train_images = []
        train_labels = []
        train_idx_map = []
        for idx, item in enumerate(self.images['train']):
            if idx in nesy_x_idxs:
                train_images.append(item[0])
                train_labels.append(item[1])
                train_idx_map.append(idx)

        new_train_ds = NSILImageLoader(train_images, train_labels, train_idx_map)

        train_loader = DataLoader(new_train_ds, batch_size=self.args.net_batch_size,
                                  num_workers=self.args.num_workers)
        test_loader = DataLoader(self.images['test'], batch_size=self.args.net_batch_size,
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
            for idx, image_idx in enumerate(list(item[:-2])):
                i = f'i{idx + 1}'
                d_list_entry[i] = self.images['train'][image_idx][0]
            data_list.append(d_list_entry)

            # Get template and create ASP Programs for obs
            t = item[-1]
            obs = self.template_manager.templates[t].get_constraints()
            obs = '\n'.join(obs)
            label = item[-2]
            obs_list.append((obs, label, t))
        return data_list, obs_list


