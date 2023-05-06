from src.NSILLogger import NSILLogger
from src.NSIL import NSIL
from global_config import set_random_seeds, SUPPROTED_NESY_REASONERS, ILP_CMD_LINE_ARGS
import argparse
import os


class NSILRun:
    def __init__(self, custom_args):
        self.nsil = None
        self.parser = argparse.ArgumentParser(description='Run NSIL with an example task')
        # NSIL Config
        nsil_group = self.parser.add_argument_group('NSIL Configuration')
        nsil_group.add_argument('--pct', type=int, default=100, help='the percentage of training data to use. default '
                                                                     '100.')
        nsil_group.add_argument('--num_workers', type=int, default=2, help='the number of workers to use in the '
                                                                           'PyTorch dataloaders.')
        nsil_group.add_argument('--tune', action='store_true', default=False, help='run hyper-parameter tuning.')
        nsil_group.add_argument('--logging', type=str, default='INFO', help='the logging level to use.')
        nsil_group.add_argument('--seed', type=int, default=5, help='the random seed to use.')
        nsil_group.add_argument('--num_iterations', type=int, default=20, help='Number of iterations to run')
        nsil_group.add_argument('--num_tuning_iterations', type=int, default=2, help='Number of tuning iterations '
                                                                                     'to run')
        nsil_group.add_argument('--explore_ex_lr', type=float, default=1, help='Between 0-1. Control the effect of '
                                                                               'the training set error on the ILP '
                                                                               'example weight penalties.')
        nsil_group.add_argument('--exploit_ex_lr', type=float, default=1, help='Between 0-1. Control the effect of '
                                                                               'the neural network confidence scores'
                                                                               'on the ILP example weight penalties.')
        nsil_group.add_argument('--save_dir', default=None, help='directory to save results. If not passed, results '
                                                                 'are saved to results/runs/run_X, where X is '
                                                                 'incremented.')
        nsil_group.add_argument('--skip_initial_testing', action='store_true', default=False, help='skip initial '
                                                                                                   'testing to save '
                                                                                                   'development or '
                                                                                                   'debugging time.')
        nsil_group.add_argument('--custom_eval_interval', type=int, default=1, help='call custom evaluation at this '
                                                                                    'interval of iterations. ')
        nsil_group.add_argument('--skip_corrective_examples', action='store_true', default=False, help='skip '
                                                                                                       'corrective '
                                                                                                       'examples')
        # Network Config
        net_group = self.parser.add_argument_group('Neural Network Configuration')
        net_group.add_argument('--nesy_reasoner', type=str, default='NeurASP', help=f'the neural-symbolic reasoning '
                                                                                    f'method to use. One of: '
                                                                                    f'{SUPPROTED_NESY_REASONERS}.')
        net_group.add_argument('--lr', type=float, default=0.001, help='learning rate for the neural network '
                                                                       'optimizer.')
        net_group.add_argument('--net_batch_size', type=int, default=64, help='the batch size to use when evaluating '
                                                                              'neural network data')
        net_group.add_argument('--num_net_epochs', type=int, default=1, help='the number of epochs to train the '
                                                                             'network for each NSIL iteration. '
                                                                             'Default 1.')
        net_group.add_argument('--save_nets', action='store_true', default=False, help='whether to save neural '
                                                                                       'networks during training')

        # ILP Config
        ilp_group = self.parser.add_argument_group('ILP Configuration')
        ilp_group.add_argument('--ilp_system', type=str, default='FastLAS', help=f'the ILP system to use. '
                                                                                 f'One of: {list(ILP_CMD_LINE_ARGS.keys())}')
        ilp_group.add_argument('--custom_ilp_cmd_line_args', default=None, help=f'custom command line arguments to '
                                                                                f'pass to the ILP system. Defaults '
                                                                                f'to: {ILP_CMD_LINE_ARGS}.')
        ilp_group.add_argument('--prune_ilp_example_weight_threshold', type=int, default=1, help='Remove ILP examples '
                                                                                                 'with weights less '
                                                                                                 'than this threshold. '
                                                                                                 'Default is 1 (no '
                                                                                                 'pruning).')
        ilp_group.add_argument('--use_bootstrap_cache', action='store_true', default=False, help='Use cached bootstrap '
                                                                                                 'hypothesis instead '
                                                                                                 'of running the '
                                                                                                 'boostrap task. '
                                                                                                 'Must have '
                                                                                                 'bootstrap_cache.las'
                                                                                                 ' file in example '
                                                                                                 'root.')
        ilp_group.add_argument('--pylasp', action='store_true', default=False, help='Run ILASP using pylasp. Requires'
                                                                                    'pylasp_run_script.py template in '
                                                                                    'the example directory.')
        ilp_group.add_argument('--ilp_config', type=str, default=None, help='key for extra ilp config. Must be present '
                                                                            'in increasing_hyp_space.py valid configs.')
        ilp_group.add_argument('--skip_symbolic_learning', action='store_true', default=False, help='don\'t run '
                                                                                                    'symbolic learning')

        if custom_args:
            self.custom_arg_group = self.parser.add_argument_group('Custom arguments for each experiment')
            for a in custom_args:
                self._add_argument(a)
        self.args = self.parse_args()
        self.logger = NSILLogger(args=self.args)
        set_random_seeds(self.args.seed)

    def _add_argument(self, a):
        if a.type == bool:
            self.custom_arg_group.add_argument(f'--{a.name}', action='store_true', default=a.default, help=a.help)
        else:
            self.custom_arg_group.add_argument(f'--{a.name}', type=a.type, default=a.default, help=a.help)

    def parse_args(self):
        args = self.parser.parse_args()
        # Perform basic validation
        assert args.nesy_reasoner in SUPPROTED_NESY_REASONERS
        assert args.num_iterations > 0
        assert args.num_tuning_iterations > 0
        assert args.num_net_epochs > 0
        assert args.ilp_system in ILP_CMD_LINE_ARGS
        if args.pylasp:
            assert args.ilp_system == 'ILASP'
        assert 0 < args.pct <= 100
        assert args.seed > -1
        assert args.net_batch_size > 0
        assert args.num_workers > 0
        assert args.logging in ['NONE', 'INFO', 'DEBUG']
        assert 0 <= args.explore_ex_lr <= 1
        assert 0 <= args.exploit_ex_lr <= 1
        assert 1 <= args.prune_ilp_example_weight_threshold <= 100
        if args.use_bootstrap_cache:
            assert os.path.exists('bootstrap_cache.las')

        # Can only run with pylasp if using ILASP
        if args.pylasp:
            assert args.ilp_system == 'ILASP'

        # During tuning, don't perform initial testing and override num iterations
        if args.tune:
            args.skip_initial_testing = True
            args.num_iterations = args.num_tuning_iterations
        return args

    def run(self, task, given_h=''):
        self.nsil = NSIL(self.args, self.logger, task)
        # For rebuttal:
        if self.args.skip_symbolic_learning:
            self.nsil.h = task.convert_las_hyp_to_nesy_r(given_h)
        self.nsil.train()

    def test(self, data_type):
        if self.nsil:
            return self.nsil.test(1, data_type)



