import shutil
from os.path import join
from pathlib import Path
from global_config import DEVICE
from datetime import datetime
import logging
import os
import json
import time
import torch

RESULT_DIR = 'results'

messages = {
    'log_save': lambda x: f'Saving log and results to: {x}',
    'device': lambda x: f'Running on device: {x}',
    'time': lambda x: f'Current machine date/time: {x}',
    'component_time': lambda x, y: f'Component: {x}, run time: {y}',
    'custom': lambda x: str(x),
    'hash_sep': lambda: '#' * 56,
    'header': lambda: 'Neural-Symbolic Inductive Learner (NSIL)',
    'test_result': lambda i, x, y, z: f'Iteration {i} test accuracy:\n\tEnd-to-end: {x}\n\tNetwork(s): {y}\n\t'
                                      f'Hypothesis: {z}',
    'nsil_train_begin': lambda: 'Starting NSIL training loop...',
    'new_iteration': lambda x: f'\n{"#"*10}\nIteration: {x}\n{"#"*10}',
    'start_ilp': lambda x: f'Running ILP system with command: \n{x}',
    'load_data': lambda: f'Loading data...',
    'generating_bootstrap': lambda: 'Generating bootstrap task...',
    'bootstrap': lambda x: f'Bootstrap task:\n{x}',
    'new_h': lambda x: f'Learned hypothesis:\n{x}\n',
    'converted_h': lambda x: f'Converted hypothesis for neural-symbolic reasoning:\n{x}\n',
    'cached_bootstrap': lambda x: f'Using Cached bootstrap hypothesis:\n{x}',
    'start_net_train': lambda x: f'Starting neural network training for {x} epoch(s)...',
    'running_net_eval': lambda x, y: f'Evaluating neural network {y} on the {x} set',
    'net_acc': lambda x, y: f'Accuracy for neural network {x}: {y}',
    'start_symbolic_eval': lambda x, y: f'Starting symbolic task evaluation for {x} set with preds type: {y}.',
    'exploration_start': lambda: 'Performing exploration... ',
    'exploitation_start': lambda: 'Performing exploitation...',
    'run_test': lambda x: f'Running testing for iteration: {x}...',
    'fnr': lambda x: f'False Negative Rate for each label/structure combination:\n{x}',
    'overwrite_results': lambda x: f'Warning: Overwriting results at {x}',
    'custom_eval': lambda: 'Performing custom evaluation...',
    'conf_mat': lambda x, y: f'Confusion matrix for network {x}:\n{y}',
    'hyper_params': lambda x: f'Best hyper-parameters obtained:\n{x}'
}


class NSILLogger:
    def __init__(self, args):
        self.args = args
        self.log_dir = self.log_setup()
        self.log_path = join(self.log_dir, 'full_log.txt')
        self.debug_path = join(self.log_dir, 'debug.txt')
        self.train_details_path = join(self.log_dir, 'train_details.txt')
        self.test_path = join(self.log_dir, 'test_log.json')
        self.custom_eval_path = join(self.log_dir, 'custom_eval.json')
        self.args_path = join(self.log_dir, 'args.json')
        self.las_tasks_dir = join(self.log_dir, 'LAS_tasks')
        self.las_cache_dir = join(self.log_dir, 'LAS_cache')
        self.networks_dir = join(self.log_dir, 'networks')
        self.pylasp_dir = join(self.log_dir, 'pylasp')
        self.times_path = join(self.log_dir, 'times.json')
        self.timer = {
            'start': -1.0,
            'iterations': {}
        }

        # Print header
        self.info('hash_sep')
        self.info('header')
        self.info('hash_sep')
        self.info('time', datetime.now())
        self.info('device', DEVICE)
        self.info('log_save', self.log_dir)
        self.save_args()

        # Create empty results files
        with open(self.test_path, 'w') as test_log:
            test_log.write(json.dumps({}))
        with open(self.custom_eval_path, 'w') as custom_f:
            custom_f.write(json.dumps({}))

        with open(self.train_details_path, 'w') as train_log:
            train_log.write('Training Details\n')
            train_log.write('For each iteration, the learned hypothesis, and run times are stored.\n')
            train_log.write('-----------------\n')
        Path(self.las_tasks_dir).mkdir()
        Path(self.las_cache_dir).mkdir()
        Path(self.networks_dir).mkdir()
        if self.args.pylasp:
            Path(self.pylasp_dir).mkdir()

    def info(self, key, *args):
        msg = messages[key](*args)
        with open(self.log_path, 'a') as logf:
            logf.write(msg)
            logf.write('\n')
        logging.info(msg)

    def debug(self, key, *args):
        msg = messages[key](*args)
        with open(self.debug_path, 'a') as debugf:
            debugf.write(msg)
            debugf.write('\n')
        logging.debug(msg)

    def save_args(self):
        self.info('custom', f'Arguments: {vars(self.args)}')
        with open(self.args_path, 'w') as argsf:
            argsf.write(json.dumps(vars(self.args), indent=2))

    def log_setup(self):
        # Configure python logger
        log_level = logging.DEBUG
        if self.args.logging == 'INFO':
            log_level = logging.INFO
        elif self.args.logging == 'NONE':
            log_level = logging.ERROR
        logging.basicConfig(level=log_level)

        # Create containing results directory if not exists
        if self.args.save_dir:
            if os.path.exists(self.args.save_dir):
                shutil.rmtree(self.args.save_dir)
            log_dir = self.args.save_dir
        else:
            exp_dir = join(RESULT_DIR, 'runs')
            Path(exp_dir).mkdir(exist_ok=True, parents=True)
            # Increment run dir
            runs = os.scandir(exp_dir)
            int_runs = [int(r.name.split('_')[1]) for r in runs if r.is_dir()]
            run_name = 'run_1'
            if len(int_runs) > 0:
                int_runs.sort()
                run_name = 'run_{0}'.format(int_runs[-1] + 1)
            log_dir = join(exp_dir, run_name)
        Path(log_dir).mkdir(parents=True)
        return log_dir

    def start_timer(self):
        self.timer['start'] = time.time()

    def add_component_time(self, iteration, key, val):
        self.timer['iterations'][iteration][key] = val
        self.info('component_time', key, val)

    def start_iteration_timer(self, i):
        self.timer['iterations'][i] = {
            'start': time.time()
        }

    def stop_iteration_timer(self, iteration, skip_time):
        # Calculate iteration time, subtract test time if required
        iteration_time = time.time() - self.timer['iterations'][iteration]['start']
        if skip_time:
            iteration_time = iteration_time - skip_time
        self.timer['iterations'][iteration]['total_without_test'] = iteration_time

    def stop_timer(self):
        overall_time = time.time() - self.timer['start']
        self.timer['overall_including_test'] = overall_time

        # Sum iteration times as overall excluding test
        overall_excl_test = 0
        for i in self.timer['iterations']:
            overall_excl_test += self.timer['iterations'][i]['total_without_test']
        self.timer['overall_excluding_test'] = overall_excl_test

        with open(self.times_path, 'w') as timesf:
            timesf.write(json.dumps(self.timer['iterations'], indent=2))

    def save_train_details_result(self, iteration, hyp):
        with open(self.train_details_path, 'a') as train_f:
            train_f.write(f'{messages["hash_sep"]()}\n')
            train_f.write(f'Iteration: {iteration}\n')
            train_f.write(f'{messages["hash_sep"]()}\n')
            train_f.write(f'Hypothesis:\n{hyp}\n')
            train_f.write(f'Timings:\n')
            train_f.write(json.dumps(self.timer['iterations'][iteration], indent=2))
            train_f.write('\n\n')

    def save_test_result(self, iteration, e_t_e_acc, nn_ac, hyp_acc, conf_mats):
        self.info('test_result', iteration, e_t_e_acc, nn_ac, hyp_acc)
        for nn in conf_mats:
            self.info('conf_mat', nn, conf_mats[nn])

        curr_results_f = open(self.test_path, 'r')
        curr_results = json.loads(curr_results_f.read())
        curr_results_f.close()

        with open(self.test_path, 'w') as test_f:
            curr_results[iteration] = {
                'end_to_end_acc': e_t_e_acc,
                'network_accuracy': nn_ac,
                'hyp_accuracy': hyp_acc
            }
            test_f.write(json.dumps(curr_results, indent=2))

    def save_LAS_task(self, t, iteration):
        with open(join(self.las_tasks_dir, f'iteration_{iteration}.las'), 'w') as tf:
            tf.write(t)

    def save_custom_eval(self, iteration, results):
        curr_results_f = open(self.custom_eval_path, 'r')
        curr_results = json.loads(curr_results_f.read())
        curr_results_f.close()
        with open(self.custom_eval_path, 'w') as custom_f:
            curr_results[iteration] = results
            custom_f.write(json.dumps(curr_results, indent=2))

    def save_pylasp_file(self, f):
        assert self.args.pylasp

        # Get current iteration and append iteration number
        iterations = os.listdir(self.pylasp_dir)
        int_iters = [int(i.split('_')[1].split('.')[0]) for i in iterations if 'iteration_' in i]
        name = 'iteration_1.las'
        if len(int_iters) > 0:
            int_iters.sort()
            name = f'iteration_{int_iters[-1] + 1}.las'
        # Save
        fpath = join(self.pylasp_dir, name)
        with open(fpath, 'w') as outf:
            outf.write(f)

    def save_network_weights(self, name, net, iteration):
        save_name = f'net_{name}_iteration_{iteration}.pt'
        save_path = join(self.networks_dir, save_name)
        torch.save(net.state_dict(), save_path)

    def save_hyper_params(self, params, name_prefix=''):
        self.info('hyper_params', params)
        if name_prefix != '':
            name_prefix = f'{name_prefix}_'
        with open(join(os.getcwd(), f'{name_prefix}hyper_params.json'), 'w') as outf:
            outf.write(json.dumps(params))


