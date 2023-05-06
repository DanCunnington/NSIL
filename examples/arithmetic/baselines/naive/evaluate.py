from ff_nsl.data import load_nn_data, load_nesy_data
from global_config import NUM_NAIVE_BASELINE_REPEATS, set_random_seeds, DEVICE
from examples.arithmetic.network import MNISTNet
from examples.recursive_arithmetic.baselines.naive.evaluate import score_network
from scipy import stats
from pathlib import Path
from os.path import join
import json
import argparse
import numpy as np
import torch


def score_repeat(repeat):
    print(f'Repeat: {repeat + 1}')
    # Score network accuracy
    net = MNISTNet(num_out=10)
    # With FF-NSL, network is same for both tasks
    if args.method == 'ff_nsl':
        net_path  = f'{args.method}/networks/{args.pct}_{repeat + 1}.pt'
    else:
        net_path = f'{args.method}/networks/{args.task_type}_{args.pct}_{repeat + 1}.pt'
    net.load_state_dict(torch.load(net_path))
    net.to(DEVICE)
    net.eval()
    n_acc, net_preds = score_network(net, nn_test_loader)
    print(n_acc)

    # Score task acc - assume correct rules
    # Load nesy test data
    _, nesy_test_data = load_nesy_data('../../data', args.task_type)

    correct = 0
    for test_ex in nesy_test_data:
        # Get nn predictions
        x_idxs = test_ex[:-1]
        y = test_ex[-1]
        ex_nn_preds = [net_preds[i] for i in x_idxs]
        if args.task_type == 'sum':
            y_pred = np.sum(np.array(ex_nn_preds))
        else:
            d1 = ex_nn_preds[0]
            d2 = ex_nn_preds[1]
            if d1 % 2 == 0:
                y_pred = d2
            else:
                y_pred = 9 + d2
        if y_pred == y:
            correct += 1
    t_acc = correct / len(nesy_test_data)
    print(t_acc)
    return n_acc, t_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', default='sum', choices=['sum', 'e9p'])
    parser.add_argument('--pct', default='100', choices=['100', '10', '5'])
    parser.add_argument('--method', default='ff_nsl', choices=['ff_nsl', 'NeurASP'])
    args = parser.parse_args()
    args.pct = int(args.pct)
    print(f'Method: {args.method}, task: {args.task_type}, pct: {args.pct}')
    # Load random seeds
    with open('../../../../seeds.txt', 'r') as sf:
        seeds = [int(s) for s in sf.readlines()]

    _, nn_test_loader = load_nn_data(args, '../../data', batch_size=8)

    # For each repeat, score network acc, task acc
    # Keep track of scores
    net_accs = []
    task_accs = []
    for r in range(NUM_NAIVE_BASELINE_REPEATS):
        set_random_seeds(seeds[r])
        net_acc, task_acc = score_repeat(r)
        net_accs.append(net_acc)
        task_accs.append(task_acc)

    # Compute averages
    avg_net_acc = np.mean(np.array(net_accs))
    std_err_net_acc = stats.sem(np.array(net_accs))
    avg_task_acc = np.mean(np.array(task_accs))
    std_err_task_acc = stats.sem(np.array(task_accs))

    # Save to file
    save_dir = f'../saved_results/{args.task_type}/{args.method}/{args.pct}'
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    with open(join(save_dir, f'results.json'), 'w') as outf:
        result_obj = {
            'network': {
                'acc': avg_net_acc,
                'std_err': std_err_net_acc
            },
            'task': {
                'acc': avg_task_acc,
                'std_err': std_err_task_acc
            }
        }
        outf.write(json.dumps(result_obj))
