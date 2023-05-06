from global_config import NUM_NAIVE_BASELINE_REPEATS, DEVICE, set_random_seeds
from examples.recursive_arithmetic.networks.Meta_Abd_CNN import Net
from examples.recursive_arithmetic.baselines.naive.ff_nsl.data import load_nn_data, load_nesy_data
from pathlib import Path
from os.path import join
from scipy import stats
import json
import torch
import argparse
import numpy as np


def score_network(n, tl):
    correct = 0
    all_preds = torch.tensor([], device='cpu')
    for data, target in tl:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = n(data)
        pred = output.argmax(dim=1, keepdim=True)
        all_preds = torch.cat((all_preds, pred.to('cpu')), 0)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(tl.dataset), all_preds


def score_test_yaml_file(net_preds, yf):
    test_preds = []
    test_targets = []
    for ex in yf:
        # Get neural network preds for images
        ex_net_preds = [net_preds[i] for i in ex.x_idxs]
        if args.task_type == 'sum':
            y_pred = np.sum(np.array(ex_net_preds))
        else:
            y_pred = np.prod(np.array(ex_net_preds))
        test_preds.append(y_pred)
        test_targets.append(ex.y)

    # Compute MAE or logMAE
    if args.task_type == 'sum':
        test_loss = torch.nn.L1Loss(reduction='sum')(
            torch.FloatTensor(test_preds), torch.FloatTensor(test_targets)).item()
    else:
        test_loss = torch.nn.L1Loss(reduction='sum')(
            torch.log(torch.FloatTensor(test_preds) + 1e-10),
            torch.log(torch.FloatTensor(test_targets) + 1e-10)).item()
    test_loss /= len(yf)
    return test_loss


def score_repeat(repeat):
    print(f'Repeat: {repeat+1}')
    # Score network accuracy
    net = Net(out_dim=10)
    net.load_state_dict(torch.load(f'{args.method}/networks/{args.task_type}_{repeat+1}.pt'))
    net.to(DEVICE)
    net.eval()
    net_acc, net_preds = score_network(net, nn_test_loader)
    print(net_acc)

    # Score task acc - assume correct rules (check rules for proof, saves calling ASP)
    # Load nesy test data
    _, nesy_test_data, extra_test = load_nesy_data('../../data', args.task_type)
    task_accs = {
        5: score_test_yaml_file(net_preds, nesy_test_data)
    }
    for etf in extra_test:
        if '10.' in etf:
            key = 10
        elif '15.' in etf:
            key = 15
        elif '100.' in etf:
            key = 100
        else:
            raise ValueError(f'Invalid test file: {etf}')
        task_accs[key] = score_test_yaml_file(net_preds, extra_test[etf])

    print(task_accs)
    print('------')
    return net_acc, task_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', default='sum', choices=['sum', 'prod'])
    parser.add_argument('--method', default='ff_nsl', choices=['ff_nsl', 'NeurASP'])
    args = parser.parse_args()
    print(f'Method: {args.method}, task: {args.task_type}')
    # Load random seeds
    with open('../../../../seeds.txt', 'r') as sf:
        seeds = [int(s) for s in sf.readlines()]

    _, nn_test_loader = load_nn_data('../../data', args.task_type, batch_size=8)

    # For each repeat, score network acc, task acc
    # Keep track of scores
    net_accs = []
    if args.task_type == 'sum':
        avg_task_accs = {
            5: [],
            10: [],
            100: []
        }
    else:
        avg_task_accs = {
            5: [],
            10: [],
            15: []
        }
    for r in range(NUM_NAIVE_BASELINE_REPEATS):
        set_random_seeds(seeds[r])
        net_acc, task_accs = score_repeat(r)
        net_accs.append(net_acc)
        for k in task_accs:
            avg_task_accs[k].append(task_accs[k])

    # Compute averages
    avg_net_acc = np.mean(np.array(net_accs))
    std_err_net_acc = stats.sem(np.array(net_accs))
    avg_task_loss = {}
    task_sem = {}
    print(f'Average Network accuracy: {avg_net_acc}')
    for k in avg_task_accs:
        avg_k_loss = np.mean(np.array(avg_task_accs[k]))
        k_sem = stats.sem(np.array(avg_task_accs[k]))
        avg_task_loss[k] = avg_k_loss
        task_sem[k] = k_sem
        print(f'Task {k} loss: {avg_k_loss}')

    # Save to file
    save_dir = f'../saved_results/{args.method}'
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    with open(join(save_dir, f'{args.task_type}.json'), 'w') as outf:
        result_obj = {
            'network': {
                'acc': avg_net_acc,
                'std_err': std_err_net_acc
            },
             'task': {
                 'loss': avg_task_loss,
                 'std_err': task_sem
             }
        }
        outf.write(json.dumps(result_obj))
