from abc import ABC, abstractmethod
from src.ILP.WCDPI import DefaultWCDPIPair
import time
import numpy as np


class NSILTask(ABC):
    def __init__(self, data, runner, net_confs, image_to_net_map, md, bk):
        """
        Class to store specific configuration for a given task
        @param data: the NSILDataset for this task
        @param runner: the NSILRun instance
        @param net_confs: dict of NSIL neural network configurations
        @param image_to_net_map: image to network mapping
        @param md: ILP mode declarations
        @param bk: ILP background knowledge
        """
        # Load datasets
        self.data = data
        self.args = runner.args
        self.logger = runner.logger
        self.net_confs = net_confs
        self.image_to_net_map = image_to_net_map
        self.nesy_data = self.data.load_nesy_data()
        self.nn_data = self.data.load_nn_data()
        self.md = md
        self.bk = bk
        self.bootstrap_examples = {}
        self.symbolic_examples = {}
        self.current_h = ''
        self.stable_model_cache = {}

    def create_LAS_task(self):
        """
        Create a LAS learning task
        @return: LAS learning task string
        """
        # Pruning via threshold argument
        def include_ex(ex):
            return ex.weight == 'inf' or ex.weight >= self.args.prune_ilp_example_weight_threshold
        example_strs = []

        # Bootstrap examples
        if len(self.bootstrap_examples) == 0:
            self.create_bootstrap_task()
        for e in self.bootstrap_examples:
            b_ex = self.bootstrap_examples[e]
            if include_ex(b_ex):
                example_strs.append(str(b_ex))

        # Corrective examples
        for e in self.symbolic_examples:
            explore = self.symbolic_examples[e].explore
            exploit = self.symbolic_examples[e].exploit
            if include_ex(explore):
                example_strs.append(str(explore))
            if include_ex(exploit):
                example_strs.append(str(exploit))
        example_strs = '\n'.join(example_strs)
        lt = f'{example_strs}\n% END_EXAMPLES\n\n% Background Knowledge\n{self.bk}\n\n% Mode Declarations\n{self.md}'
        return lt

    def update_WCDPI_weights(self, combo_id, new_weight, fnr=True):
        """
        Perform WCDPI weight updates
        @param combo_id: ID of the WCDPIPair
        @param new_weight: absolute value of the deltaW weight
        @param fnr: True if new_weight calculated from FNR, False if calculated from NN conf scores
        """
        assert combo_id in self.symbolic_examples
        assert new_weight >= 0

        def fnr_update(c, l, w):
            w = w + 1
            return int(round(c + (l*(w-c))))

        def nn_conf_update(c, l, w):
            new_w = int(round(c + (l * w)))
            if new_w < 1:
                new_w = 1
            if new_w > 101:
                new_w = 101
            return new_w

        curr_explore_weight = self.symbolic_examples[combo_id].explore.weight
        curr_exploit_weight = self.symbolic_examples[combo_id].exploit.weight
        assert curr_explore_weight != 'inf'
        assert curr_exploit_weight != 'inf'
        if fnr:
            delta_w_explore = new_weight
            delta_w_exploit = 0
            lr = self.args.explore_ex_lr
            new_explore_w = fnr_update(curr_explore_weight, lr, delta_w_explore)
            new_exploit_w = fnr_update(curr_exploit_weight, lr, delta_w_exploit)
        else:
            delta_w_explore = -new_weight
            delta_w_exploit = new_weight
            lr = self.args.exploit_ex_lr
            new_explore_w = nn_conf_update(curr_explore_weight, lr, delta_w_explore)
            new_exploit_w = nn_conf_update(curr_exploit_weight, lr, delta_w_exploit)
        self.symbolic_examples[combo_id].explore.weight = new_explore_w
        self.symbolic_examples[combo_id].exploit.weight = new_exploit_w

    def calculate_train_FNR(self, downstream_train_preds):
        """
        Calculate the False Negative Rate for each unique combination of label/structure in the task
        @return: dictionary of FNR per label/structure
        """
        combo_count = {}
        combo_correct = {}
        for idx, item in enumerate(self.nesy_data['train']):
            label = int(item[1][0].item())
            combo_id = self.get_combo_id(item)
            if combo_id:
                if combo_id not in combo_count:
                    combo_count[combo_id] = 1
                else:
                    combo_count[combo_id] += 1
                if label == downstream_train_preds[idx]:
                    if combo_id not in combo_correct:
                        combo_correct[combo_id] = 1
                    else:
                        combo_correct[combo_id] += 1
        fnr = {}
        for c in combo_count:
            if c not in combo_correct:
                correct = 0
            else:
                correct = combo_correct[c]
            fnr[c] = 100 * (1 - (correct / combo_count[c]))
        return fnr

    def exploration(self, i, downstream_train_preds, h):
        start_time = time.time()
        self.logger.info('exploration_start')
        fnr = self.calculate_train_FNR(downstream_train_preds)
        self.logger.info('fnr', fnr)
        # For each training example, create WCDPI pair weighted by FNR
        # Create one example per choice for each label/structure combo
        asp_prog = f'{self.bk}\n{h}'
        combos_done = []
        for ex in self.nesy_data['train']:
            label = ex[1][0].item()
            combo_id = self.get_combo_id(ex)
            if combo_id and combo_id not in combos_done:
                # Get all possible choices of contextual facts
                possible_ctxs = self.compute_stable_models(asp_prog, f':- not result({label}).', len(ex[0]))
                for poss in possible_ctxs:
                    # Create new explore exploit pair
                    facts = self.get_context_facts(poss)
                    ex_id = f'{combo_id}_' + '_'.join([str(p) for p in poss])
                    if ex_id not in self.symbolic_examples:
                        self.symbolic_examples[ex_id] = DefaultWCDPIPair(combo_id=ex_id,
                                                                         label=label,
                                                                         ctx_facts=[facts])
                    # Divide FNR by number of combinations
                    weight = fnr[combo_id] / len(possible_ctxs)
                    self.update_WCDPI_weights(ex_id, weight, fnr=True)
                combos_done.append(combo_id)

        total_time = time.time() - start_time
        self.logger.add_component_time(i, 'exploration', total_time)

    def exploitation(self, i, nn_output):
        start_time = time.time()
        self.logger.info('exploitation_start')
        predictions = nn_output['digit']['predictions']
        conf_scores = nn_output['digit']['confidence']
        idx_map = nn_output['digit']['idx_map']

        # Calculate aggregated conf scores for each label / structure combo and get NN predictions
        combos = {}
        for ex in self.nesy_data['train']:
            label = ex[1][0].item()
            # Get neural network predictions for the images in the nesy example
            x = [int(predictions[idx_map[i[0].item()]].item()) for i in ex[0]]
            confs = [conf_scores[idx_map[i[0].item()]].item() for i in ex[0]]
            agg_conf = np.prod(np.array(confs))
            combo_id = self.get_combo_id(ex)
            if combo_id:
                combo_id = f'{combo_id}_' + '_'.join([str(c) for c in x])
                if combo_id in combos:
                    combos[combo_id]['weights'].append(agg_conf)
                else:
                    start_list = self.get_context_facts(x)
                    combos[combo_id] = {
                        'weights': [agg_conf],
                        'context': [start_list],
                        'label': label
                    }
        for combo_id in combos:
            weight = 100 * np.mean(np.array(combos[combo_id]['weights']))
            if combo_id not in self.symbolic_examples:
                label = combos[combo_id]['label']
                ctx = combos[combo_id]['context']
                self.symbolic_examples[combo_id] = DefaultWCDPIPair(combo_id=combo_id, label=label, ctx_facts=ctx)
            self.update_WCDPI_weights(combo_id, weight, fnr=False)
        total_time = time.time() - start_time
        self.logger.add_component_time(i, 'exploitation', total_time)

    @abstractmethod
    def custom_evaluation(self, i, net_out, downstream_preds):
        """
        Perform any custom evaluation
        @param i: iteration number
        @param net_out: network predictions
        @param downstream_preds: task predictions
        """
        pass

    @abstractmethod
    def get_combo_id(self, nesy_ex, for_ilp=True):
        """
        Build a WCPDI ID from a nesy example for neural network training
        @param nesy_ex: the nesy example
        @param for_ilp: whether this call is related to the symbolic learner optimisation (true) or neural net
        training (false)
        @return: ID str
        """
        pass

    @abstractmethod
    def get_context_facts(self, nn_preds):
        """
        Build contextual symbolic facts for neural network predictions
        @param nn_preds: list of neural network predictions
        @return: list of contextual facts
        """
        pass

    @abstractmethod
    def symbolic_evaluation(self, iteration, latent_concepts, h, data_type='test'):
        """
        Evaluate the symbolic task
        @param iteration: the current NSIL iteration
        @param latent_concepts: the latent concept values for the data (either from NN or ground)
        @param data_type: training or testing data
        @param h: the current hypothesis
        @return: the final predictions on the symbolic task
        """
        pass

    @abstractmethod
    def convert_las_hyp_to_nesy_r(self, h):
        """
        Convert a LAS hypothesis into the format for the neural-symbolic reasoning system
        @param h: the LAS hypothesis
        @return: the converted h
        """
        pass

    @abstractmethod
    def model_to_network_preds(self, m):
        """
        Convert stable model into a list of network predictions
        @param m: the stable model
        @return: list of network predictionss
        """
        pass

    @abstractmethod
    def compute_stable_models(self, asp_prog, obs, num_images, for_ilp=True):
        """
        Compute stable models for NeurASP
        @param asp_prog: the ASP program of the learned hypothesis
        @param obs: label constraint from the Dataset
        @param num_images: the number of images in this example
        @param for_ilp: whether this call is related to the symbolic learner optimisation (true) or neural net
        training (false)
        @return: list of stable models
        """
        pass

    @abstractmethod
    def create_bootstrap_task(self):
        """
        Generate the bootstrap ILP learning task
        @return: the bootstrap task
        """
        pass

