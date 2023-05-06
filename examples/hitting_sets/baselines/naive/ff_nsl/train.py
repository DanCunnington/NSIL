from examples.hitting_sets.baselines.naive.ff_nsl.data import load_nn_data
from examples.recursive_arithmetic.baselines.naive.ff_nsl.train import train_net, run_tuning, score_network
from global_config import set_random_seeds, DEVICE, NUM_NAIVE_BASELINE_REPEATS
from examples.arithmetic.network import MNISTNet
from torch.optim import Adam
from os.path import join
from pathlib import Path
import torch
import argparse
import torch.nn as nn


def train_tune_evaluate(params):
    _lr = params.get("learning_rate", 0.001)
    _net = MNISTNet(num_out=5)
    # Replace softmax layer with log softmax
    _net.classifier[-1] = nn.LogSoftmax(dim=1)
    _net.to(DEVICE)
    _net.train()
    _optimizer = Adam(_net.parameters(), lr=_lr)

    for e in range(5):
        train_net(args, e, _net, _optimizer, train_loader)
    _net.eval()
    acc, _ = score_network(_net, test_loader)
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', default='hs')
    parser.add_argument('--image_type', default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--log_interval', default=50)
    parser.add_argument('--pct', default=100)
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()

    # Load image data from training set
    train_loader, test_loader = load_nn_data(args, '../../../data', batch_size=64)
    if args.tune:
        set_random_seeds(1)
        best_params = run_tuning(train_tune_evaluate)
        print(best_params)
    else:
        # Settings
        NUM_EPOCHS = 20
        if args.image_type == 'mnist':
            lr = 0.0016910558996832666
        else:
            lr = 0.001413842628230652

        # Load random seeds
        with open('../../../../../seeds.txt', 'r') as sf:
            seeds = [int(s) for s in sf.readlines()]

        for r in range(NUM_NAIVE_BASELINE_REPEATS):
            print(f'Repeat {r+1}')
            print('--------------')
            set_random_seeds(seeds[r])
            net = MNISTNet(num_out=5)
            # Replace softmax layer with log softmax
            net.classifier[-1] = nn.LogSoftmax(dim=1)
            net.to(DEVICE)
            net.train()
            optimizer = Adam(net.parameters(), lr=lr)
            for e in range(NUM_EPOCHS):
                train_net(args, e+1, net, optimizer, train_loader)

            # Save net
            Path('networks').mkdir(exist_ok=True)
            torch.save(net.state_dict(), join('networks', f'{args.image_type}_{r+1}.pt'))
