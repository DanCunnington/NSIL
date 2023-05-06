from examples.recursive_arithmetic.baselines.naive.ff_nsl.data import load_nn_data
from global_config import set_random_seeds, DEVICE, NUM_NAIVE_BASELINE_REPEATS
from examples.recursive_arithmetic.networks.Meta_Abd_CNN import Net
from torch.optim import Adam
from os.path import join
from examples.recursive_arithmetic.baselines.naive.evaluate import score_network
from ax.service.managed_loop import optimize
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn


def run_tuning(eval_fn):
    best_parameters, _, _, _ = optimize(
        parameters=[
            {"name": "learning_rate", "type": "range", "bounds": [0.0001, 0.1], "log_scale": True},
        ],
        evaluation_function=eval_fn,
        objective_name='accuracy',
    )
    return best_parameters


def train_tune_evaluate(params):
    _lr = params.get("learning_rate", 0.001)
    _momentum = params.get("momentum", 0.1)
    _net = Net(out_dim=10)
    # Replace softmax layer with log softmax
    _net.enc[-1] = nn.LogSoftmax(dim=1)
    _net.to(DEVICE)
    _net.train()
    _optimizer = Adam(_net.parameters(), lr=_lr)

    for e in range(5):
        train_net(args, e, _net, _optimizer, train_loader)
    _net.eval()
    acc, _ = score_network(_net, test_loader)
    return acc


def train_net(args, epoch, n, o, tl):
    for batch_idx, (data, target) in enumerate(tl):
        data, target = data.to(DEVICE), target.to(DEVICE)
        o.zero_grad()
        output = n(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        o.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(tl.dataset),
                100. * batch_idx / len(tl), loss.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', default='sum', choices=['sum', 'prod'])
    parser.add_argument('--log_interval', default=100)
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()

    # Load image data from training set
    train_loader, test_loader = load_nn_data('../../../data', args.task_type, batch_size=64)

    if args.tune:
        set_random_seeds(1)
        best_params = run_tuning(train_tune_evaluate)
        print(best_params)
    else:
        # Settings
        NUM_EPOCHS = 20
        # lr = 0.020776380142764502
        lr = 0.00048032996530678523
        # momentum = 0.22261281128629617

        # Load random seeds
        with open('../../../../../seeds.txt', 'r') as sf:
            seeds = [int(s) for s in sf.readlines()]

        for r in range(NUM_NAIVE_BASELINE_REPEATS):
            print(f'Repeat {r+1}')
            print('--------------')
            set_random_seeds(seeds[r])
            net = Net(out_dim=10)
            # Replace softmax layer with log softmax
            net.enc[-1] = nn.LogSoftmax(dim=1)
            net.to(DEVICE)
            net.train()
            optimizer = Adam(net.parameters(), lr=lr)
            for e in range(NUM_EPOCHS):
                train_net(args, e+1, net, optimizer, train_loader)

            # Save net
            torch.save(net.state_dict(), join('networks', f'{args.task_type}_{r+1}.pt'))
