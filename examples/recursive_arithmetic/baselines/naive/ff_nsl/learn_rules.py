from global_config import NUM_NAIVE_BASELINE_REPEATS, ILP_CMD_LINE_ARGS
from examples.recursive_arithmetic.networks.Meta_Abd_CNN import Net
from examples.recursive_arithmetic.baselines.naive.ff_nsl.data import load_nesy_data, get_mnist_train_for_rule_learning
from os.path import join
import torch
import tempfile
import argparse
import subprocess

meta_rules = {
    'm1': 'P(A, B) :- Q(A, B), m1(P, Q).',
    'm2': 'P(A, B) :- Q(A, C), P(C, B), m2(P, Q).',
    'm3': 'P(A, B) :- Q(A, C), R(C, B), m3(P, Q, R), Q != R.',
    'm4': 'P(A, B) :- Q(A, B), R(A, B), m4(P, Q, R), Q != R.',
    'm5': 'P(A) :- Q(A, B), m5(P, Q, B).',
    'm6': 'P(A) :- Q(A), m6(P, Q).',
    'm7': 'P(A, B) :- Q(A), R(A, B), m7(P, Q, R).',
    'm8': 'P(A, B) :- Q(A, B), R(B), m8(P, Q, R).'
}
meta_rule_vals = '\n'.join(meta_rules.values())

bk = '''
% List definition in ASP
list(L) :- start_list(L).
list(T) :- list((_, T)).
head(L, H) :- list(L), L = (H, _).
tail(L, T) :- list(L), L = (_, T).
empty(empty).

% Arithmetic Knowledge - add, mult, eq
add(L, (X+Y, T)) :- list(L), L = (X, (Y, T)).
list(L) :- add(_, L).
mult(L, (X*Y, T)) :- list(L), L = (X, (Y, T)).
list(L) :- mult(_, L).
eq(L, ELT) :- list(L), L = (ELT, empty).

% Link learned f to result, ensure only one result
result(R) :- start_list(L), f(L, R).
:- result(X), result(Y), X < Y.
'''

md = f'''
#predicate(base, head/2).
#predicate(base, tail/2).
#predicate(base, add/2).
#predicate(base, mult/2).
#predicate(base, eq/2).
#predicate(base, empty/1).
#predicate(target, f/2).

% Meta rules
{meta_rule_vals}

#modem(2, m1(target/2, any/2)).
#modem(2, m2(target/2, any/2)).
#modem(3, m3(target/2, any/2, any/2)).
#modem(3, m4(target/2, any/2, any/2)).
#modem(2, m5(target/1, any/2)).
#modem(2, m6(target/1, any/1)).
#modem(3, m7(target/2, any/1, any/2)).
#modem(3, m8(target/2, any/2, any/1)).
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', default='sum', choices=['sum', 'prod'])
    args = parser.parse_args()
    print('Loading data...')
    train_data, _, _ = load_nesy_data('../../../data', args.task_type)
    mnist_train = get_mnist_train_for_rule_learning()

    # For each repeat
    for r in range(NUM_NAIVE_BASELINE_REPEATS):
        print(f'Repeat: {r+1}')

        # Load pre-trained network
        net = Net(out_dim=10)
        net.load_state_dict(torch.load(join('networks', f'{args.task_type}_{r+1}.pt')))
        net.eval()

        examples = ''
        # Run forward pass on images in nesy train file
        ex_id = 0
        for ex in train_data:
            image_idxs = ex.x_idxs
            predictions = [int(torch.argmax(net(mnist_train[i][0].unsqueeze(0))).item()) for i in image_idxs]

            # Create example for ILASP with noise penalty 10
            sl_str = ''
            for idx, p in enumerate(predictions):
                if idx == 0:
                    sl_str += f'({p}, empty)'
                else:
                    sl_str = f'({p}, {sl_str})'
            sl_str = f'start_list({sl_str}).'
            symbolic_ex = f'#pos(ex_{ex_id}@10, {{ result({ex.y}) }}, {{ }}, {{\n'
            symbolic_ex += f'\t:- result(X), X != {ex.y}.\n'
            symbolic_ex += f'\t{sl_str}\n'
            symbolic_ex += '}).\n'
            examples += symbolic_ex
            ex_id += 1

        # Create a temporary file for ILASP task with examples, bk, md
        task = f'{examples} \n {bk} \n {md}'

        # Run ILASP and save rules
        temp_task = tempfile.NamedTemporaryFile(prefix='NSIL_tmp_file_')
        temp_task.write(task.encode())
        ilp_system = 'ILASP'
        cmd_line_args = ILP_CMD_LINE_ARGS[ilp_system]
        cmd_line_args += ' --quiet'
        BASE_DIR = '../../../../../'
        ilp_exec = join(BASE_DIR, ilp_system)
        cmd = f'{ilp_exec} {temp_task.name} {cmd_line_args}'
        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=True)

        output = result.stdout.decode()

        # Parse the output
        output = "\n".join([ll.rstrip() for ll in output.splitlines() if ll.strip() and ll != ""])

        # Close temporary file
        temp_task.close()

        with open(join('rules', f'{args.task_type}_{r+1}.las'), 'w') as rule_f:
            rule_f.write(output)
