from os.path import abspath, join, dirname
from ax.service.managed_loop import optimize
import torch
import numpy as np
import random
import multiprocessing

NUM_CPU = multiprocessing.cpu_count()
SUPPROTED_NESY_REASONERS = ['NeurASP']
ILP_CMD_LINE_ARGS = {
    'FastLAS': '--solver="ASP" --fl2',
    'ILASP': '--version=4 --strict-types --restarts --max-rule-length=3'
}

IMAGE_DATA_DIR = join(dirname(abspath(__file__)), 'data')
HYP_START_ID = '%NSIL_HYP_START'
HYP_END_ID = '%NSIL_HYP_END'
NUM_NAIVE_BASELINE_REPEATS = 5

# Get device
# Running on CPU for now due to the following error when calling backward.:
'''
NotImplementedError: The operator 'aten::logical_and.out' is not current implemented for the MPS device. 
If you want this op to be added in priority during the prototype phase of this feature, please comment on 
https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable 
`PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running 
natively on MPS.
'''
if torch.__version__ == '1.12.0' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device('cpu')
    torch.set_default_dtype(torch.float32)
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')


def set_random_seeds(seed=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_seeds():
    seeds = set()
    while len(seeds) < 20:
        seeds.add(random.randint(0, 10000))
    with open('seeds.txt', 'w') as sf:
        for s in seeds:
            sf.write(str(s) + '\n')


class CustomArgument:
    def __init__(self, a_name, a_type, a_default, a_help):
        self.name = a_name
        self.type = a_type
        self.default = a_default
        self.help = a_help


class CustomILPConfig:
    def __init__(self, bk, md):
        self.bk = bk
        self.md = md


# Tune learning rate and momentum for network SGD optimiser
def run_tuning(eval_fn):
    best_parameters, _, _, _ = optimize(
        parameters=[
            {"name": "learning_rate", "type": "range", "bounds": [0.0001, 0.1], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        ],
        evaluation_function=eval_fn,
        objective_name='accuracy',
    )
    return best_parameters


# Generate random seeds
if __name__ == "__main__":
    generate_seeds()
