from src.NSILRun import NSILRun
from RecursiveArithmeticTask import RecursiveArithmeticTask
from RecursiveArithmeticData import RecursiveArithmeticData
from examples.recursive_arithmetic.networks.Meta_Abd_CNN import Net
from torch import optim
from global_config import CustomArgument, run_tuning
from src.NSILNetworkConfiguration import NSILNetworkConfiguration


def train_tune_evaluate(params):
    _lr = params.get("learning_rate", 0.001)
    _momentum = params.get("momentum", 0.1)
    _optimizer = optim.SGD(net.parameters(), lr=_lr, momentum=_momentum)
    _net_confs = {net_name: NSILNetworkConfiguration(name=net_name, net=net, num_out=num_net_ouptuts, optim=_optimizer)}
    _task = RecursiveArithmeticTask(data, nsil, _net_confs, image_to_network_map,
                                    digit_vals=list(range(0, num_net_ouptuts)))
    # Run NSIL
    nsil.run(_task)

    # Meta_Abd data has no validation set so we use the training set
    _, acc = nsil.test(data_type='train')
    return acc


if __name__ == '__main__':
    # Setup custom arguments
    which_task = CustomArgument(a_name='task_type', a_type=str, a_default='sum', a_help='sum or product task')
    meta_abd_data_arg = CustomArgument(a_name='meta_abd_data', a_type=bool, a_default=False, a_help='whether to use the'
                                                                                                    ' first 3000 '
                                                                                                    'samples for '
                                                                                                    'training as '
                                                                                                    'Meta_Abd does.')
    ilp_max_len = CustomArgument(a_name='ilp_max_example_len', a_type=int, a_default=4, a_help='the maximum example '
                                                                                               'length to use within '
                                                                                               'the corrective examples'
                                                                                               'for the ILP task.')
    net_max_len = CustomArgument(a_name='net_max_example_len', a_type=int, a_default=5, a_help='the maximum example '
                                                                                               'length to use in neural'
                                                                                               ' network training.')

    nsil = NSILRun(custom_args=[meta_abd_data_arg, ilp_max_len, net_max_len, which_task])
    assert nsil.args.net_max_example_len > 0
    assert nsil.args.ilp_max_example_len > 0
    assert nsil.args.task_type in ['sum', 'prod']

    # Setup data loaders
    train_f = f'my{nsil.args.task_type}_full.yaml'
    test_f = f'my{nsil.args.task_type}_full_test.yaml'
    data = RecursiveArithmeticData(nsil, root_dir='data', train_file=train_f, test_file=test_f)

    # Setup networks and optimizers
    num_net_ouptuts = 10
    net_name = 'digit'
    net = Net(out_dim=num_net_ouptuts)

    # Assign each image to a network
    image_to_network_map = {
        'i1': net_name,
        'i2': net_name,
        'i3': net_name,
        'i4': net_name,
        'i5': net_name
    }

    # Run tuning or experiments
    if nsil.args.tune:
        best_params = run_tuning(train_tune_evaluate)
        nsil.logger.save_hyper_params(best_params, name_prefix=nsil.args.task_type)

    else:
        if nsil.args.task_type == 'sum':
            lr = 0.001030608122509068
            momentum = 0.02784393448382616
        else:
            lr = 0.00010722517722295946
            momentum = 0.7977277142927051
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        net_confs = {net_name: NSILNetworkConfiguration(name=net_name, net=net, num_out=num_net_ouptuts,
                                                        optim=optimizer)}

        # Setup recursive task
        task = RecursiveArithmeticTask(data, nsil, net_confs, image_to_network_map,
                                       digit_vals=list(range(0, num_net_ouptuts)))

        # Run NSIL
        nsil.run(task)


