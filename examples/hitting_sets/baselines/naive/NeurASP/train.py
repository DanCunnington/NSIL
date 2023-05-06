from examples.recursive_arithmetic.baselines.naive.NeurASP.train import TmpLogger
from examples.recursive_arithmetic.baselines.naive.evaluate import score_network
from examples.hitting_sets.HittingSetsData import HittingSetsData
from examples.hitting_sets.HittingSetsTask import HittingSetsTask
from examples.hitting_sets.baselines.naive.evaluate import templates
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
    parser.add_argument('--task_type', default='hs', choices=['hs', 'chs'])
    parser.add_argument('--image_type', default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--pct', default=100)
    parser.add_argument('--num_net_epochs', default=1)
    parser.add_argument('--meta_abd_data', default=False)
    parser.add_argument('--num_workers', default=2)
    parser.add_argument('--net_batch_size', default=64)
    args = parser.parse_args()

    tmp_logger = TmpLogger()
    tmp_runner = {'args': args, 'logger': tmp_logger}
    tmp_runner = Namespace(**tmp_runner)

    # Setup data loaders
    data = HittingSetsData(tmp_runner, root_dir='../../../data', templates=templates)

    # Load random seeds
    with open('../../../../../seeds.txt', 'r') as sf:
        seeds = [int(s) for s in sf.readlines()]

    for r in range(NUM_NAIVE_BASELINE_REPEATS):
        print(f'Repeat: {r + 1}')
        set_random_seeds(seeds[r])

        # Setup networks and optimizers
        num_net_ouptuts = 5
        net_name = 'digit'
        net = MNISTNet(num_out=num_net_ouptuts)

        # Assign each image to a network
        image_to_network_map = {
            'i1': net_name,
            'i2': net_name,
            'i3': net_name,
            'i4': net_name,
        }

        # Hyper-params
        lr = 0.0008253878514942516
        momentum = 0.7643463721907878
        if args.image_type == 'fashion_mnist':
            lr = 0.0021947689485666017
            momentum = 0.0

        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        net_confs = {net_name: NSILNetworkConfiguration(name=net_name, net=net, num_out=num_net_ouptuts,
                                                        optim=optimizer)}

        # Setup arithmetic task
        task = HittingSetsTask(data, tmp_runner, net_confs, image_to_network_map,
                               digit_vals=list(range(1, num_net_ouptuts + 1)),
                               ilp_config=None)

        nasp = NeurASP(args, tmp_logger, task)

        # Train with correct rules
        if args.task_type == 'hs':
            h = '''
             :- ss_element(V1,V2); not hit(V1); elt(V2); ss(V1).
             0 {hs(V1,V2) } 1 :- elt(V2); hs_index(V1).
             hit(V1) :- hs(V3,V2); ss_element(V1,V2); hs_index(V3); elt(V2); ss(V1).
             :- hs(V3,V1); hs(V3,V2); V1 != V2; hs_index(V3); elt(V2); elt(V1).'''
        else:
            h = '''
             :- ss_element(3,V2); ss_element(V1,1); elt(V2); ss(V1).
             :- ss_element(V1,V2); not hit(V1); elt(V2); ss(V1).
             0 {hs(V1,V2) } 1 :- elt(V2); hs_index(V1).
             hit(V1) :- hs(V3,V2); ss_element(V1,V2); hs_index(V3); elt(V2); ss(V1).
             :- hs(V3,V1); hs(V3,V2); V1 != V2; hs_index(V3); elt(V2); elt(V1).'''
        nn_data = data.load_nn_data()
        NUM_EPOCHS = 20
        for e in range(NUM_EPOCHS):
            nasp.train(h)
            _n = nasp.net_confs[net_name].net
            # Score
            acc, _ = score_network(_n, nn_data[net_name]['test'])
            print(acc)

        Path('networks').mkdir(exist_ok=True)

        # Save resulting network
        _n = nasp.net_confs[net_name].net
        torch.save(_n.state_dict(), join('networks', f'{args.task_type}_{args.image_type}_{r + 1}.pt'))
