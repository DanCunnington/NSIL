from examples.hitting_sets.baselines.naive.ff_nsl.data import load_nn_data, load_nesy_data
from global_config import NUM_NAIVE_BASELINE_REPEATS, set_random_seeds, DEVICE
from examples.arithmetic.network import MNISTNet
from examples.recursive_arithmetic.baselines.naive.evaluate import score_network
from examples.hitting_sets.baselines.naive.score_hitting_sets import decide_hitting_set
from scipy import stats
from pathlib import Path
from os.path import join
import json
import argparse
import numpy as np
import torch

templates = [
        [[-1], [-1], [-1], [-1]],
        [[-1, -1], [-1], [-1]],
        [[-1], [-1, -1], [-1]],
        [[-1], [-1], [-1, -1]],
        [[-1], [-1, -1, -1]],
        [[-1, -1], [-1, -1]],
        [[-1, -1, -1], [-1]],
        [[-1, -1, -1, -1]]
    ]


def score_repeat(repeat):
    print(f'Repeat: {repeat + 1}')
    # Score network accuracy
    net = MNISTNet(num_out=5)
    # With FF-NSL, network is same for both tasks
    if args.method == 'ff_nsl':
        net_path = f'{args.method}/networks/{args.image_type}_{repeat + 1}.pt'
    else:
        net_path = f'{args.method}/networks/{args.task_type}_{args.image_type}_{repeat + 1}.pt'
    net.load_state_dict(torch.load(net_path, map_location='cpu'))
    net.to(DEVICE)
    net.eval()
    n_acc, net_preds = score_network(net, nn_test_loader)
    print(n_acc)

    # Score task acc - assume correct rules
    # Load nesy test data
    _, nesy_test_data = load_nesy_data('../../data', args.task_type, args.image_type)

    correct = 0
    for test_ex in nesy_test_data:
        # Get nn predictions
        x_idxs = test_ex[:-2]
        template = test_ex[-1]
        y = test_ex[-2]
        ex_nn_preds = [int(net_preds[i][0].item()) for i in x_idxs]

        # Calculate y_pred depending on task
        # Otherwise calculate standard hitting sets
        y_pred = decide_hitting_set(ex_nn_preds, template, templates, '../../data', args)
        # For CHS y is 0 if >= 3 subsets and 1 exists anywhere in collection
        if args.task_type == 'chs':
            ex_nn_preds = [i+1 for i in ex_nn_preds]
            if len(templates[template]) >= 3 and 1 in ex_nn_preds:
                y_pred = 0

        if y_pred == y:
            correct += 1
    t_acc = correct / len(nesy_test_data)
    print(t_acc)
    return n_acc, t_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', default='hs', choices=['hs', 'chs'])
    parser.add_argument('--image_type', default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--pct', default=100)
    parser.add_argument('--method', default='ff_nsl', choices=['ff_nsl', 'NeurASP'])
    args = parser.parse_args()
    print(f'Method: {args.method}, task: {args.task_type}, images: {args.image_type}')
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
    save_dir = f'../saved_results/{args.task_type.upper()}_{args.image_type}/{args.method}'
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
