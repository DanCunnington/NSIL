from src.NeuralComponent import NeuralComponent
from src.ILP.ILPSystem import ILP
from sklearn.metrics import confusion_matrix
import time


class NSIL:
    def __init__(self, args, logger, task):
        """
        Class to manage NSIL training
        """
        self.args = args
        self.logger = logger
        self.task = task
        self.neural = NeuralComponent(args, logger, task)
        self.symbolic = ILP(args, logger)
        self.h = None

    def train(self, test_during=True):
        self.logger.start_timer()
        self.logger.info('nsil_train_begin')

        for i in range(1, self.args.num_iterations+1):
            self.logger.info('new_iteration', i)
            self.logger.start_iteration_timer(i)

            # Run the ILP task
            if not self.args.skip_symbolic_learning:
                h = self._run_ILP(i)
                # Convert h
                self.h = self.task.convert_las_hyp_to_nesy_r(h)

            # Initially, run testing
            # Don't add the initial test time to the overall learning time
            skip_time = None
            if i == 1 and not self.args.skip_initial_testing:
                # Pass iteration 0 to indicate initial test before training begins
                skip_time, _ = self.test(0)

            # Train the neural network using neural-symbolic reasoning
            self.neural.train(i, self.h)

            # Get NN output on training set
            nn_out = self.neural.get_network_output(data_type='train')

            # Make downstream symbolic predictions
            downstream_preds = self.task.symbolic_evaluation(i, nn_out, self.h, data_type='train')

            # Perform exploration and exploitation
            if not self.args.skip_corrective_examples:
                self.task.exploration(i, downstream_preds, self.h)
                self.task.exploitation(i, nn_out)

            # Save iteration time
            self.logger.stop_iteration_timer(i, skip_time=skip_time)
            self.logger.save_train_details_result(i, self.h)

            # Run testing
            if test_during:
                self.test(i)

    def test(self, i, data_type='test'):
        start_time = time.time()
        self.logger.info('run_test', i)

        # Get neural network predictions and accuracy
        net_out, net_acc = self.neural.evaluate(data_type=data_type)

        # Get predictions on downstream task when neural network predictions are used, and also ground 100%
        # accurate neural network predictions to evaluate the learned hypothesis
        downstream_preds = self.task.symbolic_evaluation(i, net_out, self.h, data_type=data_type)

        # Copy net out and swap predictions for ground labels
        _ground_net_out = {}
        conf_mats = {}
        for nn in net_out:
            _ground_net_out[nn] = {
                'predictions': net_out[nn]['labels'],
                'idx_map': net_out[nn]['idx_map']
            }
            # Also get confusion matrix for each network
            conf_mats[nn] = str(confusion_matrix(net_out[nn]['labels'], net_out[nn]['predictions']))
        downstream_preds_for_h_eval = self.task.symbolic_evaluation(i, _ground_net_out, self.h, data_type=data_type,
                                                                    preds_type='ground')
        e_t_e_acc, h_acc = self._downstream_task_eval(downstream_preds, downstream_preds_for_h_eval,
                                                      data_type=data_type)

        # Save to log
        self.logger.save_test_result(i, e_t_e_acc, net_acc, h_acc, conf_mats)

        # Perform any task specific evaluation
        if i > 0 and i % self.args.custom_eval_interval == 0:
            self.task.custom_evaluation(i, net_out, downstream_preds)
        total_time = time.time() - start_time
        return total_time, e_t_e_acc

    def _downstream_task_eval(self, nn_preds, ground_preds, data_type='test'):
        """
        Calculate downstream task accuracy
        @param nn_preds: downstream task predictions when the neural network(s) are used to predict input features
        @param ground_preds: downstream task predictions when ground 100% accurate neural network predictions are used
        @param data_type: which dataset to use. train, val, or test
        @return: tuple of accuracy values
        """
        nn_preds_correct = 0
        ground_preds_correct = 0
        total = 0
        for idx, row in enumerate(self.task.nesy_data[data_type]):
            label = int(row[1][0].item())
            if label == nn_preds[idx]:
                nn_preds_correct += 1
            if label == ground_preds[idx]:
                ground_preds_correct += 1
            total += 1
        nn_acc = nn_preds_correct / total
        ground_acc = ground_preds_correct / total
        return nn_acc, ground_acc

    def _run_ILP(self, iteration):
        if iteration == 1:
            # Return the cached hypothesis if required
            if self.args.use_bootstrap_cache:
                # Create the task to ensure any examples are setup correctly
                self.task.create_bootstrap_task()

                # Return the cached hypothesis
                with open('bootstrap_cache.las', 'r') as bfile:
                    bootstrap_hyp = bfile.read()
                    self.logger.info('cached_bootstrap', bootstrap_hyp)
                    return bootstrap_hyp
            else:
                las_task = self.task.create_bootstrap_task()
        else:
            las_task = self.task.create_LAS_task()
        self.logger.save_LAS_task(las_task, iteration)
        h, ilp_time = self.symbolic.run(las_task, iteration)
        self.logger.add_component_time(iteration, 'ILP', ilp_time)
        return h


