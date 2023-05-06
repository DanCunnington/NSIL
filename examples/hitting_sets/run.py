from src.NSILRun import NSILRun
from HittingSetsTask import HittingSetsTask
from HittingSetsData import HittingSetsData
from examples.arithmetic.network import MNISTNet
from torch import optim
from global_config import CustomArgument, run_tuning
from src.NSILNetworkConfiguration import NSILNetworkConfiguration
from examples.hitting_sets.increasing_hyp_space import extra_configs
from torchvision.models import vgg16
import torch.nn as nn


def train_tune_evaluate(params):
    _lr = params.get("learning_rate", 0.001)
    _momentum = params.get("momentum", 0.1)
    _optimizer = optim.SGD(net.parameters(), lr=_lr, momentum=_momentum)
    _net_confs = {net_name: NSILNetworkConfiguration(name=net_name, net=net, num_out=num_net_ouptuts, optim=_optimizer)}
    _task = HittingSetsTask(data, nsil, _net_confs, image_to_network_map, digit_vals=list(range(1, num_net_ouptuts+1)))

    # Run NSIL
    nsil.run(_task)
    _, acc = nsil.test(data_type='val')
    return acc


if __name__ == '__main__':
    # Setup custom arguments
    which_task = CustomArgument(a_name='task_type', a_type=str, a_default='hs', a_help='hs or chs task')
    which_images = CustomArgument(a_name='image_type', a_type=str, a_default='mnist', a_help='mnist, fashion_mnist, '
                                                                                             'or cifar-10 images.')

    nsil = NSILRun(custom_args=[which_task, which_images])
    assert nsil.args.task_type in ['hs', 'chs']
    assert nsil.args.image_type in ['mnist', 'fashion_mnist', 'cifar_10']

    # Setup possible hititng sets templates
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
    data = HittingSetsData(nsil, root_dir='data', templates=templates)

    # Setup networks and optimizers
    num_net_ouptuts = 5
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
        'i3': net_name,
        'i4': net_name,
    }

    # Run tuning or experiments
    if nsil.args.tune:
        best_params = run_tuning(train_tune_evaluate)
        nsil.logger.save_hyper_params(best_params, name_prefix=nsil.args.image_type)
    else:
        # Manually override hyper-parameters
        lr = 0.0008253878514942516
        momentum = 0.7643463721907878
        if nsil.args.image_type == 'fashion_mnist':
            lr = 0.0021947689485666017
            momentum = 0.0
        elif nsil.args.image_type == 'cifar_10':
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

        # Setup hitting sets task
        task = HittingSetsTask(data, nsil, net_confs, image_to_network_map,
                               digit_vals=list(range(1, num_net_ouptuts+1)),
                               ilp_config=ilp_conf)

        # Run NSIL
        nsil.run(task)
