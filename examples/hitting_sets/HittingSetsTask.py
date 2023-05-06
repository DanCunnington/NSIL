from src.NSILTask.base import NSILTask
from src.ILP.WCDPI import WCDPI, MetaLevelInjectionBinaryWCDPIPair
from src.ASPUtils import find_all_stable_models, batch_run_evaluation
from global_config import HYP_START_ID, HYP_END_ID
import time
import re
import json
import itertools
import numpy as np
import sys


class HittingSetsTask(NSILTask):
    def __init__(self, data, runner, net_confs, image_to_net_map, digit_vals, ilp_config=None):
        """
        Hitting Sets Task
        @param data: NSILDataset
        @param runner: NSILRun instance
        @param net_confs: dict of NSIL neural network configurations
        @param image_to_net_map: image to network map
        @param digit_vals: possible digit values (e.g. [0,1,...,9])
        @param ilp_config: custom bk and md
        """
        self.digit_vals = digit_vals
        self.template_manager = data.template_manager

        if ilp_config:
            bk = ilp_config.bk
            md = ilp_config.md
        else:
            bk = '''
            ss(1..4).
            hs_index(1..2).
            elt(1..5).
            '''

            md = '''
            #inject("
            1 { example_active(EG) : group(G, EG) } :- group(G, _).
            :~ not example_active(EG), weight(EG, W).[W@1, eg_weight, W, EG]
            :- #count { H : nge_HYP(H) } > 5.
            ").
            #modeha(hs(var(hs_index), var(elt))).
            #modeb(hs(var(hs_index), var(elt)),(positive)).
            #modeb(var(elt) != var(elt)).
            #modeb(ss_element(var(ss), var(elt)),(positive)).
            #modeh(hit(var(ss))).
            #modeb(hit(var(ss))).
            #bias(":- not lb(0), choice_rule.").
            #bias(":- not ub(1), choice_rule.").
            #bias(":- in_head(H1), in_head(H2), H1<H2.").
            
            #modeb(ss_element(3, var(elt)), (positive)).
            #modeb(ss_element(var(ss), 1), (positive)).
            '''

        # Remove spaces introduced by multi-line string
        def clean(x): return '\n'.join([l.strip() for l in x.split('\n')])

        md = clean(md)
        bk = clean(bk)
        super().__init__(data, runner, net_confs, image_to_net_map, md, bk)

    def create_pylasp_file(self, weights):
        # Create PyLASP script and save to run_dir
        pylasp_template = 'pylasp_run_script.py'
        pylasp_f = open(pylasp_template, 'r').read()
        pylasp_f = pylasp_f.replace('<<NSIL_WEIGHTS_TO_REPLACE>>', json.dumps(weights))
        self.logger.save_pylasp_file(pylasp_f)

    def create_LAS_task(self):
        """
        Create a LAS learning task
        @return: LAS learning task string
        """
        # Pruning via threshold argument
        def include_ex(ex):
            return ex.weight == 'inf' or ex.weight >= self.args.prune_ilp_example_weight_threshold
        example_strs = []
        all_weights = {}

        # Bootstrap examples
        if len(self.bootstrap_examples) == 0:
            self.create_bootstrap_task()
        for e in self.bootstrap_examples:
            b_ex = self.bootstrap_examples[e]
            if include_ex(b_ex):
                all_weights[e] = b_ex.weight
                example_strs.append(str(b_ex))

        # Corrective examples
        for e in self.symbolic_examples:
            pos = self.symbolic_examples[e].pos
            neg = self.symbolic_examples[e].neg
            if include_ex(pos):
                example_strs.append(str(pos))
                all_weights[pos.ex_id] = pos.weight
            if include_ex(neg):
                example_strs.append(str(neg))
                all_weights[neg.ex_id] = neg.weight

        self.create_pylasp_file(all_weights)
        example_strs = '\n'.join(example_strs)
        lt = f'{example_strs}\n% END_EXAMPLES\n\n% Background Knowledge\n{self.bk}\n\n% Mode Declarations\n{self.md}'
        return lt

    def _create_n_choice_rules(self, n=4):
        """
        Create choice rules for each digit fact
        @param n: number of digits
        @return: choice rules
        """
        cr = []
        for d in range(1, n+1):
            _d_rule = '1 {'
            _d_rule += '; '.join([f'digit({d},{v})' for v in self.digit_vals])
            _d_rule = f'{_d_rule} }} 1.'
            cr.append(_d_rule)
        return cr

    def model_to_network_preds(self, m):
        m = sorted(m.split(' '))
        # Extract network predictions from stable model
        net_preds = [int(f.split(f'digit({idx + 1},')[1].split(')')[0]) for idx, f in enumerate(m)]
        return net_preds

    def post_process_models(self, models, ctx_constr):
        # Create arrays for digit choices
        # Sort digits into sets
        processed_models = []
        for digit_choice in models:
            ss_element_rules = '\n'.join([r for r in ctx_constr if 'ss_element' in r])

            for idx, d in enumerate(digit_choice):
                target = f'X) :- digit({idx+1},X)'
                ss_element_rules = ss_element_rules.replace(target, str(d-1)+')')
            sets = []
            for r in [s for s in ss_element_rules.split('\n') if 'ss_element' in s]:
                ss_id = int(r.split('ss_element(')[1].split(',')[0])
                ss_val = int(r.split(',')[1].split(')')[0])
                if len(sets) < ss_id:
                    sets.append([ss_val])
                else:
                    sets[ss_id - 1].append(ss_val)

            # Check for duplicate sets
            no_dup_sets = True
            for s in sets:
                if sets.count(s) > 1:
                    no_dup_sets = False

            # Check no items in each set are duplicated
            no_dup_items = True
            for s in sets:
                if len(set(s)) != len(s):
                    no_dup_items = False

            # Check items in each set are sorted in numerical order
            set_items_sorted = True
            for s in sets:
                if sorted(s) != s:
                    set_items_sorted = False

            # Check subsets are sorted by first digit in each subset
            first_digit_each_set = []
            for s in sets:
                first_digit_each_set.append(s[0])

            subsets_sorted = False
            if sorted(first_digit_each_set) == first_digit_each_set:
                subsets_sorted = True

            if no_dup_sets and no_dup_items and set_items_sorted and subsets_sorted:
                processed_models.append(sets)
        return processed_models

    def _get_digit_choices(self, t):
        # Get constraints
        constr = t.get_constraints()

        # Get digit choice rules
        cr = '\n'.join(self._create_n_choice_rules())
        _ctx_constr = '\n'.join(constr)
        prog = f'{_ctx_constr}\n{cr}\n#show digit/2.'

        # Find stable models
        models = find_all_stable_models(prog, self.model_to_network_preds)
        models = self.post_process_models(models, constr)
        return models

    def create_ss_element_facts_from_sets(self, c, increment=True):
        rules = ''
        for idx, subset in enumerate(c):
            r = f'ss_element({idx+1},'
            for el in subset:
                if increment:
                    el = el + 1
                rules += f'{r}{el}).\n\t'
        return rules

    def create_bootstrap_task(self):
        # For each template, create LAS examples with possible choices
        all_weights = {}
        for t in self.template_manager.templates:
            poss_labels = self.template_manager.get_labels_for_template(t)
            groups = self.template_manager.get_group_ids_for_template(t)
            # If there is only one possible label per template, create a single WCDPI with infinite weight for all
            # possibilities
            possible_digit_combos = self._get_digit_choices(t)
            if len(poss_labels) == 1:
                group_id = groups[0]
                for combo in possible_digit_combos:
                    ctx = self.create_ss_element_facts_from_sets(combo)
                    label = poss_labels[0]
                    digits = list(itertools.chain(*combo))
                    digit_combo = '_'.join([str(d+1) for d in digits])
                    ex_id = f'template_{t.t_id}_{digit_combo}'
                    if label == 1:
                        positive = True
                    else:
                        positive = False
                    weight = 'inf'
                    all_weights[ex_id] = weight
                    self.bootstrap_examples[ex_id] = WCDPI(ex_id, positive, weight, [], [], [ctx], group=group_id)
            else:
                # Otherwise create a pair of WCDPIs with adjustable weights
                for combo in possible_digit_combos:
                    ctx = self.create_ss_element_facts_from_sets(combo)
                    digits = list(itertools.chain(*combo))
                    digit_combo_id = '_'.join([str(d+1) for d in digits])
                    combo_id = f'template_{t.t_id}_{digit_combo_id}'
                    new_pair = MetaLevelInjectionBinaryWCDPIPair(combo_id, groups, [ctx])
                    # Add weights for pylasp
                    new_pair_weights = new_pair.get_example_ids_and_weights()
                    for e in new_pair_weights:
                        all_weights[e[0]] = e[1]
                    self.symbolic_examples[combo_id] = new_pair

        # Create final task
        self.create_pylasp_file(all_weights)
        ex_str = "\n".join([str(self.bootstrap_examples[e]) for e in self.bootstrap_examples]) + '\n'
        ex_str += "\n".join([str(self.symbolic_examples[e]) for e in self.symbolic_examples])
        return f'{ex_str}\n{self.bk}\n{self.md}'

    def convert_las_hyp_to_nesy_r(self, h):
        """
        Convert a LAS hypothesis into a NeurASP representation
        @param h: the hypothesis
        """
        converted_h = h
        self.logger.info('converted_h', converted_h)
        return converted_h

    def get_combo_id(self, ex, for_ilp=True):
        label = ex[1][0].item()
        template = ex[3][0].item()
        return f'template_{template}_label_{label}'

    def get_context_facts(self, nn_preds):
        """
        Skip this because we impelment our own version...
        """
        return

    def flatten(self, models):
        return [list(itertools.chain.from_iterable(m)) for m in models]

    def compute_stable_models(self, asp_prog, obs, num_images, for_ilp=True):
        obs_prog, label, template = obs

        # Check cache
        combo_id = f'template_{template}_label_{label}'
        prog_id = re.sub(r"[\n\t\s]*", "", asp_prog)
        if prog_id in self.stable_model_cache and label in self.stable_model_cache[prog_id]:
            return self.stable_model_cache[prog_id][combo_id]

        # Create choice rules
        cr = self._create_n_choice_rules()
        cr = '\n'.join(cr)

        # For hitting sets, obs contains ASP program and label
        split_obs_prog = obs_prog.split('\n')

        # If it's a positive example, we use the learned rules
        show_rule = '#show digit/2.'
        program = f'{cr}\n{asp_prog}\n{obs_prog}\n{show_rule}'
        models = find_all_stable_models(program, self.model_to_network_preds)
        models = self.flatten(self.post_process_models(models, split_obs_prog))
        if label == 0:
            # Otherwise, we compute the stable models without using rules first
            # and then subtract the stable models when the rules are added
            asp_prog_without_rules = asp_prog.split(f'\n{HYP_START_ID}\n')[0] + asp_prog.split(f'{HYP_END_ID}\n')[1]
            prog_without_rules = f'{cr}\n{asp_prog_without_rules}\n{obs_prog}\n{show_rule}'
            full_models = find_all_stable_models(prog_without_rules, self.model_to_network_preds)
            full_models = self.flatten(self.post_process_models(full_models, split_obs_prog))
            diff = [x for x in full_models if x not in models]
            models = diff
        # Save to cache
        if prog_id not in self.stable_model_cache:
            self.stable_model_cache[prog_id] = {combo_id: models}
        else:
            self.stable_model_cache[prog_id][combo_id] = models
        return models

    def symbolic_evaluation(self, i, latent_concepts, h, data_type='test', preds_type='nn'):
        self.logger.info('start_symbolic_eval', data_type, preds_type)
        start_time = time.time()
        lc = latent_concepts['digit']['predictions']
        idx_map = latent_concepts['digit']['idx_map']
        # Create clingo task and evaluate with learned hypothesis
        header = f'{self.bk}\n{h}'
        footer = ''
        examples = []

        # Create examples
        for ex in self.nesy_data[data_type]:
            # On the training set, map the NeSy image idx to the prediction idx
            # On this task add 1 to convert to digits 1-5
            if data_type == 'train' and idx_map:
                x = [lc[idx_map[i[0].item()]] for i in ex[0]]
            else:
                x = [lc[i[0].item()] for i in ex[0]]
            t_id = ex[-1][0].item()
            facts = self.template_manager.templates[t_id].get_ss_element_facts_with_digits(x)
            examples.append(facts)

        # Batch run evaluation
        predictions = batch_run_evaluation(header, examples, footer)

        # Assume prediction is true if there are models, otherwise false
        predictions = [len(predictions[key]) > 0 for key in sorted(predictions.keys())]
        end_time = time.time() - start_time
        if data_type == 'train':
            self.logger.add_component_time(i, f'symbolic_{data_type}_{preds_type}_preds_eval', end_time)
        return predictions

    def custom_evaluation(self, i, net_out, downstream_preds):
        return

    def exploration(self, i, downstream_train_preds, h):
        """
        Implement custom exploration for this task due to positive/negative examples
        """
        start_time = time.time()
        self.logger.info('exploration_start')
        fnr = self.calculate_train_FNR(downstream_train_preds)
        self.logger.info('fnr', fnr)
        # For each training example, create WCDPI pair weighted by FNR
        # Create one example per choice for each label/structure combo
        combos_done = []
        # Get all possible choices of digits with learned hyp
        cr = '\n'.join(self._create_n_choice_rules())
        asp_prog = f'{self.bk}{cr}\n{HYP_START_ID}\n{h}\n{HYP_END_ID}\n'

        for ex in self.nesy_data['train']:
            label = ex[1][0].item()
            t_id = ex[3][0].item()

            template_id = f'template_{t_id}'
            combo_id = f'{template_id}_label_{label}'
            t = self.template_manager.templates[t_id]
            group_ids = self.template_manager.get_group_ids_for_template(t)
            # Only perform exploration if there are multiple groups
            if len(group_ids) > 1 and combo_id and combo_id not in combos_done:
                constr = t.get_constraints()
                _ctx_constr = '\n'.join(constr)
                # Build obs tuple
                obs = (_ctx_constr, label, t_id)
                models = self.compute_stable_models(asp_prog, obs, num_images=4)
                for m in models:
                    # Update weights
                    ex_id = f'{template_id}_' + '_'.join([str(d+1) for d in m])
                    if ex_id not in self.symbolic_examples:
                        print(f'ERROR: {ex_id} does not exist during exploration.')
                        sys.exit(1)
                    self._set_explore_exploit(ex_id, label)

                    # Divide FNR by number of combinations
                    weight = fnr[combo_id] / len(models)
                    self.update_WCDPI_weights(ex_id, weight, fnr=True)
                combos_done.append(combo_id)

        total_time = time.time() - start_time
        self.logger.add_component_time(i, 'exploration', total_time)

    def exploitation(self, i, nn_output):
        """
        Custom implementation for hitting sets
        """
        start_time = time.time()
        self.logger.info('exploitation_start')
        predictions = nn_output['digit']['predictions']
        conf_scores = nn_output['digit']['confidence']
        idx_map = nn_output['digit']['idx_map']

        # Calculate aggregated conf scores for each label / structure combo and get NN predictions
        combos = {}
        for ex in self.nesy_data['train']:
            label = ex[1][0].item()
            t_id = ex[3][0].item()

            # Get neural network predictions for the images in the nesy example
            x = [int(predictions[idx_map[i[0].item()]].item()) for i in ex[0]]
            confs = [conf_scores[idx_map[i[0].item()]].item() for i in ex[0]]
            agg_conf = np.prod(np.array(confs))
            combo_id = f'template_{t_id}_' + '_'.join([str(c+1) for c in x])
            if combo_id in combos:
                combos[combo_id]['weights'].append(agg_conf)
            else:
                ctx = self.template_manager.templates[t_id].get_ss_element_facts_with_digits(x)
                combos[combo_id] = {
                    'weights': [agg_conf],
                    'context': [ctx],
                    'label': label,
                    'template': t_id
                }
        for combo_id in combos:
            weight = 100 * np.mean(np.array(combos[combo_id]['weights']))
            label = combos[combo_id]['label']

            # Only update weights during exploitation, and only if more than one possible label
            t = self.template_manager.templates[combos[combo_id]['template']]
            if combo_id in self.symbolic_examples and len(self.template_manager.get_labels_for_template(t)) > 1:
                self._set_explore_exploit(combo_id, label)
                self.update_WCDPI_weights(combo_id, weight, fnr=False)
        total_time = time.time() - start_time
        self.logger.add_component_time(i, 'exploitation', total_time)

    def _set_explore_exploit(self, ex_id, label):
        # Set explore and exploit to pos or neg depending on label
        if label == 1:
            self.symbolic_examples[ex_id].explore = self.symbolic_examples[ex_id].neg
            self.symbolic_examples[ex_id].exploit = self.symbolic_examples[ex_id].pos
        else:
            self.symbolic_examples[ex_id].explore = self.symbolic_examples[ex_id].pos
            self.symbolic_examples[ex_id].exploit = self.symbolic_examples[ex_id].neg

