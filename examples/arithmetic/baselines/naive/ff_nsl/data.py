from examples.arithmetic.ArithmeticData import get_data
from torchvision import transforms
from global_config import IMAGE_DATA_DIR
from torchvision.datasets import MNIST
from src.NSILDataset import NSILImageLoader
from torch.utils.data import DataLoader
from os.path import join
import pandas as pd


def load_nesy_data(root_dir, task_type):
    train_csv = pd.read_csv(join(root_dir, f'{task_type}', 'train.csv')).values
    test_csv = pd.read_csv(join(root_dir, f'{task_type}', 'test.csv')).values
    return train_csv, test_csv


def load_nn_data(args, root_dir, batch_size):
    train_csv_file = pd.read_csv(f'{root_dir}/{args.task_type}/train.csv')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image_data = {
        'train': MNIST(root=IMAGE_DATA_DIR, train=True, download=True, transform=transform),
        'test': MNIST(root=IMAGE_DATA_DIR, train=False, download=True, transform=transform)
    }

    # Remove images from MNIST training set that aren't in nesy data
    nesy_train = get_data(args, train_csv_file, train=True)
    x_idxs = [list(i[:-1]) for i in nesy_train]

    # Flatten
    nesy_x_idxs = [item for sublist in x_idxs for item in sublist]

    # Extract train_images
    train_images = []
    train_labels = []
    train_idx_map = []
    for idx, item in enumerate(image_data['train']):
        if idx in nesy_x_idxs:
            train_images.append(item[0])
            train_labels.append(item[1])
            train_idx_map.append(idx)

    new_train_ds = NSILImageLoader(train_images, train_labels, train_idx_map)
    train_loader = DataLoader(new_train_ds, batch_size=batch_size,
                              num_workers=2)
    test_loader = DataLoader(image_data['test'], batch_size=batch_size,
                             num_workers=2)
    return train_loader, test_loader

