from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import pandas as pd
from os.path import exists, join
from pathlib import Path
import pickle


class NSILImageLoader(Dataset):
    def __init__(self, images, labels, idx_map):
        """
        Custom image loader for neural network data
        @param images: list of image tensors
        @param labels: list of image labels
        @param idx_map: mapping from each idx 0-len(images) to the image ID in the NeSy data file.
        """
        self.images = images
        self.labels = labels
        self.idx_map = idx_map
        self.reverse_idx_map = {i: idx for (idx, i) in enumerate(self.idx_map)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class NSILDataset(ABC):
    def __init__(self, runner, root_dir='.'):
        self.args = runner.args
        self.logger = runner.logger
        self.root_dir = root_dir
        self.NeurASP_data = self.cache_NeurASP_data()

    def cache_NeurASP_data(self):
        """
        Save or Load NeurASP data to the cache
        @return: data_list, obs_list
        """
        # Always convert NeurASP data for now...
        cache_name = f'{self.args.pct}_pct'
        if hasattr(self.args, 'image_type'):
            cache_name += f'_{self.args.image_type}'
        if hasattr(self.args, 'task_type'):
            cache_name += f'_{self.args.task_type}'
        if hasattr(self.args, 'meta_abd_data'):
            if self.args.meta_abd_data:
                cache_name += '_meta_abd'
        cache_name += '.pickle'
        cache_dir = 'data_cache'
        cache_path = join(cache_dir, cache_name)
        Path(cache_dir).mkdir(exist_ok=True)
        if exists(cache_path):
            # Load from cache
            with open(cache_path, 'rb') as cachef:
                return pickle.load(cachef)
        else:
            # Convert and save to cache
            neurasp_data = self.convert_to_NeurASP()
            with open(cache_path, 'wb') as cachef:
                pickle.dump(neurasp_data, cachef)
            return neurasp_data

    @abstractmethod
    def convert_to_NeurASP(self):
        """
        Convert data to NeurASP format
        @return: data_list and obs_list
        """
        pass

    @abstractmethod
    def load_nesy_data(self):
        """
        Load neural-symbolic data with image idxs, downstream labels, and ground image labels
        @return: dict of training validation, and test data loaders, e.g.:
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        """
        pass

    @abstractmethod
    def load_nn_data(self):
        """
        Load data for the neural network
        @return: dict of the form nn_data = {
            '<net_name>': {
                'train': train_loader,
                'test': test_loader
            }
        }
        """
        pass
