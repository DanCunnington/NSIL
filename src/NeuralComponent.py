from src.NeuralSymbolicReasoner.selector import api as nesy_r_api
from global_config import DEVICE
from tqdm import tqdm
import torch
import time


class NeuralComponent:
    def __init__(self, args, logger, task):
        self.args = args
        self.logger = logger
        self.task = task
        self.net_data = self.task.data.load_nn_data()

        # Setup neural-symbolic reasoning
        self.nesy_r = nesy_r_api(method=self.args.nesy_reasoner)(args, logger, task)

    def train(self, it, h):
        """
        Train the neural network using neural-symbolic reasoning
        :param it: the iteration number
        :type it: int
        :param h: the candidate hypothesis
        :type h: str
        """
        self.logger.info('start_net_train', self.args.num_net_epochs)

        # Set networks to training mode
        for nn in self.task.net_confs:
            self.task.net_confs[nn].net.train()
        start_time = time.time()
        self.nesy_r.train(h)
        total_time = time.time() - start_time
        self.logger.add_component_time(it, 'nn_training', total_time)

        # Save network weights
        if self.args.save_nets or it == self.args.num_iterations:
            for nn in self.task.net_confs:
                net = self.task.net_confs[nn].net
                self.logger.save_network_weights(nn, net, it)

    def evaluate(self, data_type='test'):
        # Network images are from the training set if using validation data
        if data_type == 'val':
            data_type = 'train'

        # Call get network output and score against ground-truth data, for each network type
        all_net_outs = self.get_network_output(data_type)
        all_net_accs = {}
        for nn in self.net_data:
            net_out = all_net_outs[nn]
            preds = net_out['predictions']
            labels = net_out['labels']
            correct = preds.eq(labels.view_as(preds)).sum().item()
            acc = correct / len(labels)
            self.logger.info('net_acc', nn, acc)
            all_net_accs[nn] = acc
        return all_net_outs, all_net_accs

    def get_network_output(self, data_type):
        # Set networks to eval mode
        results = {}
        for nn in self.task.net_confs:
            assert data_type in self.net_data[nn]
            loader = self.net_data[nn][data_type]
            self.task.net_confs[nn].net.eval()
            self.logger.info('running_net_eval', data_type, nn)

            with torch.no_grad():
                all_preds = torch.tensor([], device='cpu')
                all_confs = torch.tensor([], device='cpu')
                all_labels = torch.tensor([], device='cpu')
                for data, targets in tqdm(loader, ncols=50):
                    data, targets = data.to(DEVICE), targets.to(DEVICE)
                    outputs = self.task.net_confs[nn].net(data)
                    confs, preds = torch.max(outputs, 1)

                    all_preds = torch.cat((all_preds, preds.to('cpu')), 0)
                    all_confs = torch.cat((all_confs, confs.to('cpu')), 0)
                    all_labels = torch.cat((all_labels, targets.to('cpu')), 0)

                # Get idx map for prediction conversion
                idx_map = None
                if data_type == 'train':
                    idx_map = self.net_data[nn]['train'].dataset.reverse_idx_map
                results[nn] = {
                    'predictions': all_preds,
                    'confidence': all_confs,
                    'labels': all_labels,
                    'idx_map': idx_map
                }
        return results



