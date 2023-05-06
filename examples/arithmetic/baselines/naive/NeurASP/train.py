from examples.recursive_arithmetic.baselines.naive.NeurASP.train import TmpLogger
from examples.recursive_arithmetic.baselines.naive.evaluate import score_network
from examples.arithmetic.ArithmeticData import ArithmeticData
from examples.arithmetic.ArithmeticTask import ArithmeticTask
from examples.arithmetic.network import MNISTNet
from src.NSILNetworkConfiguration import NSILNetworkConfiguration
from src.NeuralSymbolicReasoner.NeurASP import NeurASP
from argparse import Namespace
from global_config import set_random_seeds, NUM_NAIVE_BASELINE_REPEATS
from torch import optim
from pathlib import Path
from os.path import join
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', default='sum', choices=['sum', 'e9p'])
    parser.add_argument('--pct', default='100', choices=['100', '10', '5'])
    parser.add_argument('--num_net_epochs', default=1)
    parser.add_argument('--meta_abd_data', default=False)
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--net_batch_size', default=64)
    args = parser.parse_args()
    args.pct = int(args.pct)

    tmp_logger = TmpLogger()
    tmp_runner = {'args': args, 'logger': tmp_logger}
    tmp_runner = Namespace(**tmp_runner)

    # Setup data loaders
    data = ArithmeticData(tmp_runner, root_dir='../../../data', task=args.task_type, images='mnist')

    # Load random seeds
    with open('../../../../../seeds.txt', 'r') as sf:
        seeds = [int(s) for s in sf.readlines()]

    for r in range(NUM_NAIVE_BASELINE_REPEATS):
        print(f'Repeat: {r + 1}')
        set_random_seeds(seeds[r])

        # Setup networks and optimizers
        num_net_ouptuts = 10
        net_name = 'digit'
        net = MNISTNet(num_out=num_net_ouptuts)

        # Assign each image to a network
        image_to_network_map = {
            'i1': net_name,
            'i2': net_name,
        }

        # Hyper-params
        lr = 0.0005404844707369889
        momentum = 0.446859063384064
        if args.task_type == 'e9p':
            lr = 0.0021947689485666017
            momentum = 0

        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        net_confs = {net_name: NSILNetworkConfiguration(name=net_name, net=net, num_out=num_net_ouptuts,
                                                        optim=optimizer)}

        # Setup arithmetic task
        task = ArithmeticTask(data, tmp_runner, net_confs, image_to_network_map, ilp_config=None,
                              digit_vals=list(range(0, num_net_ouptuts)))

        nasp = NeurASP(args, tmp_logger, task)

        # Train with correct rules
        if args.task_type == 'sum':
            h = 'solution(V0,V1,V2) :- V2 = V1 + V0, digit_type(V0), digit_type(V1), num(V2).'
        else:
            h = '''
            solution(V0,V1,V2) :- V2 = V1, even(V0), digit_type(V0), digit_type(V1), num(V2).
            solution(V0,V1,V2) :- not even(V0), plus_nine(V1,V2), digit_type(V0), digit_type(V1), num(V2).'''
        nn_data = data.load_nn_data()
        NUM_EPOCHS = 20
        for e in range(NUM_EPOCHS):
            nasp.train(h)
            _n = nasp.net_confs[net_name].net
            # Score
            acc, _ = score_network(_n, nn_data[net_name]['test'])
            print(acc)

        Path('networks').mkdir(exist_ok=True)

        # Get resulting network
        _n = nasp.net_confs[net_name].net
        torch.save(_n.state_dict(), join('networks', f'{args.task_type}_{args.pct}_{r + 1}.pt'))
