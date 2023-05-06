from src.NSILRun import NSILRun
from ArithmeticTask import ArithmeticTask
from ArithmeticData import ArithmeticData
from examples.arithmetic.network import MNISTNet
from torch import optim
from global_config import CustomArgument, run_tuning
from src.NSILNetworkConfiguration import NSILNetworkConfiguration
from examples.arithmetic.increasing_hyp_space import extra_configs
from torchvision.models import vgg16
import torch.nn as nn


def train_tune_evaluate(params):
    _lr = params.get("learning_rate", 0.001)
    _momentum = params.get("momentum", 0.1)
    _optimizer = optim.SGD(net.parameters(), lr=_lr, momentum=_momentum)
    _net_confs = {net_name: NSILNetworkConfiguration(name=net_name, net=net, num_out=num_net_ouptuts, optim=_optimizer)}
    _task = ArithmeticTask(data, nsil, _net_confs, image_to_network_map,
                           digit_vals=list(range(0, num_net_ouptuts)))
    # Run NSIL
    nsil.run(_task)
    _, acc = nsil.test(data_type='val')
    return acc


if __name__ == '__main__':
    # Setup custom arguments
    which_task = CustomArgument(a_name='task_type', a_type=str, a_default='sum', a_help='sum or e9p task')
    which_image = CustomArgument(a_name='image_type', a_type=str, a_default='mnist', a_help='mnist or cifar_10')

    nsil = NSILRun(custom_args=[which_task, which_image])
    assert nsil.args.task_type in ['sum', 'e9p']
    assert nsil.args.image_type in ['mnist', 'cifar_10']

    # Setup data loaders
    data = ArithmeticData(nsil, root_dir='data', task=nsil.args.task_type, images=nsil.args.image_type)

    # Setup networks and optimizers
    num_net_ouptuts = 10
    net_name = 'digit'
    if nsil.args.image_type == 'cifar_10':
        net = vgg16(num_classes=num_net_ouptuts)
        net.classifier = nn.Sequential(
            *net.classifier,
            nn.Softmax(1),
        )
    else:
        net = MNISTNet(num_out=num_net_ouptuts)

    # Assign each image to a network
    image_to_network_map = {
        'i1': net_name,
        'i2': net_name,
    }

    # Run tuning or experiments
    if nsil.args.tune:
        best_params = run_tuning(train_tune_evaluate)
        nsil.logger.save_hyper_params(best_params, name_prefix=nsil.args.task_type)
    else:
        # Hyper-params
        lr = 0.0005404844707369889
        momentum = 0.446859063384064
        if nsil.args.task_type == 'e9p':
            lr = 0.0021947689485666017
            momentum = 0

        if nsil.args.image_type == 'cifar_10':
            lr = 0.001
            momentum = 0
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        net_confs = {net_name: NSILNetworkConfiguration(name=net_name, net=net, num_out=num_net_ouptuts,
                                                        optim=optimizer)}

        # Load in custom background knowledge and mode declarations if required
        ilp_conf = nsil.args.ilp_config
        if ilp_conf:
            assert ilp_conf in extra_configs
            ilp_conf = extra_configs[ilp_conf]

        # Setup arithmetic task
        task = ArithmeticTask(data, nsil, net_confs, image_to_network_map, ilp_config=ilp_conf,
                              digit_vals=list(range(0, num_net_ouptuts)))
        # Run NSIL
        nsil.run(task)


