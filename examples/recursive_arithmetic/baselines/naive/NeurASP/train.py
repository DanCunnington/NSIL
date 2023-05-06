from examples.recursive_arithmetic.RecursiveArithmeticData import RecursiveArithmeticData
from examples.recursive_arithmetic.RecursiveArithmeticTask import RecursiveArithmeticTask
from src.NSILNetworkConfiguration import NSILNetworkConfiguration
from argparse import Namespace
from examples.recursive_arithmetic.networks.Meta_Abd_CNN import Net
from src.NeuralSymbolicReasoner.NeurASP import NeurASP
from torch import optim
from global_config import NUM_NAIVE_BASELINE_REPEATS, set_random_seeds
from os.path import join
from pathlib import Path
from examples.recursive_arithmetic.baselines.naive.evaluate import score_network
import torch
import argparse


class TmpLogger:
    def info(self, *_args):
        print(*_args)


def score():
    _n = nasp.net_confs[net_name].net
    _n.eval()
    # Score
    acc, _ = score_network(_n, nn_data[net_name]['test'])
    print(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', default='sum', choices=['sum', 'prod'])
    parser.add_argument('--num_net_epochs', default=1)
    parser.add_argument('--pct', default=100)
    parser.add_argument('--meta_abd_data', default=True)
    parser.add_argument('--net_max_example_len', default=5)
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--net_batch_size', default=64)
    args = parser.parse_args()
    tmp_logger = TmpLogger()
    tmp_runner = {'args': args, 'logger': tmp_logger}
    tmp_runner = Namespace(**tmp_runner)

    # Setup data loaders
    train_f = f'my{args.task_type}_full.yaml'
    test_f = f'my{args.task_type}_full_test.yaml'
    data = RecursiveArithmeticData(tmp_runner, root_dir='../../../data', train_file=train_f, test_file=test_f)

    # Load random seeds
    with open('../../../../../seeds.txt', 'r') as sf:
        seeds = [int(s) for s in sf.readlines()]

    for r in range(NUM_NAIVE_BASELINE_REPEATS):
        print(f'Repeat: {r+1}')
        set_random_seeds(seeds[r])
        # Setup networks and optimizers
        num_net_ouptuts = 10
        net_name = 'digit'
        net = Net(out_dim=num_net_ouptuts)

        # Assign each image to a network
        image_to_network_map = {
            'i1': net_name,
            'i2': net_name,
            'i3': net_name,
            'i4': net_name,
            'i5': net_name
        }
        if args.task_type == 'sum':
            lr = 0.001030608122509068
            momentum = 0.02784393448382616
        else:
            lr = 0.00010722517722295946
            momentum = 0.7977277142927051

        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        net_confs = {net_name: NSILNetworkConfiguration(name=net_name, net=net, num_out=num_net_ouptuts,
                                                        optim=optimizer)}

        # Setup recursive task
        task = RecursiveArithmeticTask(data, tmp_runner, net_confs, image_to_network_map,
                                       digit_vals=list(range(0, num_net_ouptuts)))

        nasp = NeurASP(args, tmp_logger, task)

        # Train with correct rules
        if args.task_type == 'sum':
            h = '''
            f(A, B) :- eq(A, B).
            f(A, B) :- add(A, C), f(C, B).
            '''
        else:
            h = '''
            f(A, B) :- eq(A, B).
            f(A, B) :- mult(A, C), f(C, B).
            '''
        nn_data = data.load_nn_data()
        NUM_EPOCHS = 20
        score()
        for e in range(NUM_EPOCHS):
            _n = nasp.net_confs[net_name].net
            _n.train()
            nasp.train(h)
            score()

        Path('networks').mkdir(exist_ok=True)

        # Get resulting network
        _n = nasp.net_confs[net_name].net
        torch.save(_n.state_dict(), join('networks', f'{args.task_type}_{r + 1}.pt'))

