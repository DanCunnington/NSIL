from os.path import join
from torchvision import transforms
from global_config import IMAGE_DATA_DIR
from torchvision.datasets import MNIST
from src.NSILDataset import NSILImageLoader
from torch.utils.data import DataLoader
import yaml


def get_data(yaml_file, train_image=False, train_rules=False):
    if train_image:
        num_examples = 3000
    elif train_rules:
        num_examples = 3100
    else:
        num_examples = len(yaml_file)
    if train_rules:
        data = yaml_file[3000:num_examples]
    else:
        data = yaml_file[:num_examples]
    return data


def load_nesy_data(root_dir, task_type):
    train_file = f'my{task_type}_full.yaml'
    test_file = f'my{task_type}_full_test.yaml'
    if 'sum' in test_file:
        extra_test_files = [
            'mysum_full_test_10.yaml',
            'mysum_full_test_100.yaml'
        ]
    else:
        extra_test_files = [
            'myprod_full_test_10.yaml',
            'myprod_full_test_15.yaml'
        ]
    with open(join(root_dir, train_file), 'r') as f:
        train_yaml = yaml.load(f, Loader=yaml.Loader)

    with open(join(root_dir, test_file), 'r') as f:
        test_yaml = yaml.load(f, Loader=yaml.Loader)

    train_yaml = get_data(train_yaml, train_rules=True)
    extra_test_yamls = {}
    for yf in extra_test_files:
        with open(join(root_dir, yf), 'r') as f:
            extra_test_yamls[yf] = yaml.load(f, Loader=yaml.Loader)
    return train_yaml, test_yaml, extra_test_yamls


def get_mnist_train_for_rule_learning():
    MNIST_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = MNIST(root=IMAGE_DATA_DIR, train=True, download=False, transform=MNIST_transform)
    return mnist_train


def load_nn_data(root_dir, task_type, batch_size):
    train_file = f'my{task_type}_full.yaml'

    with open(join(root_dir, train_file), 'r') as f:
        train_yaml = yaml.load(f, Loader=yaml.Loader)

    MNIST_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    MNIST_data = {
        'train': MNIST(root=IMAGE_DATA_DIR, train=True, download=False, transform=MNIST_transform),
        'test': MNIST(root=IMAGE_DATA_DIR, train=False, download=False, transform=MNIST_transform)
    }

    # Remove images from MNIST training set that aren't in nesy data
    nesy_train = get_data(train_yaml, train_image=True)
    x_idxs = [i.x_idxs for i in nesy_train]
    nesy_x_idxs = [item for sublist in x_idxs for item in sublist]

    # Extract train_images
    train_images = []
    train_labels = []
    train_idx_map = []
    for idx, item in enumerate(MNIST_data['train']):
        if idx in nesy_x_idxs:
            train_images.append(item[0])
            train_labels.append(item[1])
            train_idx_map.append(idx)

    new_train_ds = NSILImageLoader(train_images, train_labels, train_idx_map)

    train_loader = DataLoader(new_train_ds, batch_size=batch_size,
                              num_workers=2)
    test_loader = DataLoader(MNIST_data['test'], batch_size=batch_size,
                             num_workers=2)
    return train_loader, test_loader
