from src.NSILTask.base import NSILTask
from src.ILP.WCDPI import WCDPI
from src.ASPUtils import find_all_stable_models, batch_run_evaluation
from tqdm import tqdm
import string
import time
import torch
import math
import re
import numpy as np


class RecursiveArithmeticTask(NSILTask):
    def __init__(self, data, runner, net_confs, image_to_net_map, digit_vals):
        """
        Recursive Arithmetic Tasks
        @param data: NSILDataset
        @param runner: NSILRun instance
        @param net_confs: dict of NSIL neural network configurations
        @param image_to_net_map: image to network map
        @param digit_vals: possible digit values (e.g. [0,1,...,9])
        """
        self.digit_vals = digit_vals
        # Note - meta_rules must contain spaces between variables, e.g., P, Q, R
        self.meta_rules = {
            'm1': 'P(A, B) :- Q(A, B), m1(P, Q).',
            'm2': 'P(A, B) :- Q(A, C), P(C, B), m2(P, Q).',
            'm3': 'P(A, B) :- Q(A, C), R(C, B), m3(P, Q, R), Q != R.',
            'm4': 'P(A, B) :- Q(A, B), R(A, B), m4(P, Q, R), Q != R.',
            'm5': 'P(A) :- Q(A, B), m5(P, Q, B).',
            'm6': 'P(A) :- Q(A), m6(P, Q).',
            'm7': 'P(A, B) :- Q(A), R(A, B), m7(P, Q, R).',
            'm8': 'P(A, B) :- Q(A, B), R(B), m8(P, Q, R).'
        }
        meta_rule_vals = '\n'.join(self.meta_rules.values())

        bk = '''
        % List definition in ASP
        list(L) :- start_list(L).
        list(T) :- list((_, T)).
        head(L, H) :- list(L), L = (H, _).
        tail(L, T) :- list(L), L = (_, T).
        empty(empty).

        % Arithmetic Knowledge - add, mult, eq
        add(L, (X+Y, T)) :- list(L), L = (X, (Y, T)).
        list(L) :- add(_, L).
        mult(L, (X*Y, T)) :- list(L), L = (X, (Y, T)).
        list(L) :- mult(_, L).
        eq(L, ELT) :- list(L), L = (ELT, empty).
        
        % Link learned f to result, ensure only one result
        result(R) :- start_list(L), f(L, R).
        :- result(X), result(Y), X < Y.
        '''

        md = f'''
        #predicate(base, head/2).
        #predicate(base, tail/2).
        #predicate(base, add/2).
        #predicate(base, mult/2).
        #predicate(base, eq/2).
        #predicate(base, empty/1).
        #predicate(target, f/2).
        
        % Meta rules
        {meta_rule_vals}
        
        #modem(2, m1(target/2, any/2)).
        #modem(2, m2(target/2, any/2)).
        #modem(3, m3(target/2, any/2, any/2)).
        #modem(3, m4(target/2, any/2, any/2)).
        #modem(2, m5(target/1, any/2)).
        #modem(2, m6(target/1, any/1)).
        #modem(3, m7(target/2, any/1, any/2)).
        #modem(3, m8(target/2, any/2, any/1)).
        '''

        # Remove spaces introduced by multi-line string
        def clean(x): return '\n'.join([l.strip() for l in x.split('\n')])
        md = clean(md)
        bk = clean(bk)
        super().__init__(data, runner, net_confs, image_to_net_map, md, bk)
        self.extra_test_nesy_data = self.data.load_extra_test_nesy_data()

    def _create_n_choice_rules(self, n):
        """
        Create choice rules for each digit fact
        @param n: number of digits
        @return: choice rules
        """
        cr = []
        for d in range(n):
            _d_rule = '1 {'
            _d_rule += '; '.join([f'digit({d+1},{v})' for v in self.digit_vals])
            _d_rule = f'{_d_rule} }} 1.'
            cr.append(_d_rule)
        return cr

    def create_bootstrap_task(self):
        choice_rules = {}
        # Lists are minimum length 2 up to max specified by user
        bootstrap_list_lens = list(range(2, self.args.ilp_max_example_len+1))
        for l in bootstrap_list_lens:
            choice_rules[l] = self._create_n_choice_rules(l)

        # For each unique_label in the dataset, if list length <= n, create LAS example with choice rules
        # for possible digit values
        # Firstly, get the unique labels that satisfy the list length
        unique_labels_for_lens = {}
        for item in self.nesy_data['train']:
            x_idxs = item[0]
            y = item[1]
            list_len = len(x_idxs)
            if list_len in bootstrap_list_lens:
                if list_len not in unique_labels_for_lens:
                    unique_labels_for_lens[list_len] = []
                unique_labels_for_lens[list_len].append(y[0].item())

        # Secondly, create examples with choice rules
        ex_id = 0
        for l in bootstrap_list_lens:
            poss_labels = set(unique_labels_for_lens[l])
            for pl in poss_labels:
                self.bootstrap_examples[f'bootstrap_{ex_id}'] = WCDPI(
                    ex_id=f'bootstrap_{ex_id}',
                    positive=True,
                    weight='inf',
                    inclusion=[f'result({pl})'],
                    exclusion=[],
                    context=choice_rules[l] + [self._create_start_list_choice(l)]
                )
                ex_id += 1

        # Create final task
        ex_str = "\n".join([str(self.bootstrap_examples[e]) for e in self.bootstrap_examples])
        return f'{ex_str}\n{self.bk}\n{self.md}'

    def convert_las_hyp_to_nesy_r(self, h):
        """
        Convert a LAS hypothesis into a NeurASP representation
        @param h: the hypothesis
        """
        # Replace meta-rules accordingly
        rules = h.split('\n')
        converted_rules = []
        for r in rules:
            if r != '':
                m_n, atoms = r.split('(')
                atoms = atoms.split(')')[0].split(',')
                meta_rule = self.meta_rules[m_n]
                without_m, m_vars = meta_rule.split(f', {m_n}')

                # Replace P, Q, R, etc with atoms in learned rules
                m_vars = m_vars.split('(')[1].split(')')[0].split(', ')
                for idx, v in enumerate(m_vars):
                    without_m = without_m.replace(v, atoms[idx])
                without_m += '.'
                converted_rules.append(without_m)
        converted_h = '\n'.join(converted_rules)
        self.logger.info('converted_h', converted_h)
        return converted_h

    def model_to_network_preds(self, m):
        m = sorted(m.split(' '))
        # Extract network predictions from stable model
        net_preds = [int(f.split(f'digit({idx+1},')[1].split(')')[0]) for idx, f in enumerate(m)]
        return net_preds

    def get_combo_id(self, ex, for_ilp=True):
        list_len = len(ex[0])
        label = ex[1][0].item()
        # Check if example is less than or equal to permitted length
        if for_ilp:
            max_len = self.args.ilp_max_example_len
        else:
            max_len = self.args.net_max_example_len
        if list_len <= max_len:
            return f'list_len_{list_len}_label_{label}'
        else:
            return None

    def _get_combo_id_num_images_label(self, num_images, label, for_ilp=True):
        _list = list(range(num_images))
        _label = [torch.tensor(label)]
        return self.get_combo_id([_list, _label], for_ilp)

    def _create_start_list_choice(self, list_len):
        start_list = 'start_list('
        digit_facts = ''
        for idx, n in enumerate(range(list_len)):
            letter = string.ascii_uppercase[idx]
            start_list += f'({letter}, '
            digit_facts += f'digit({idx + 1}, {letter}), '
        start_list += f'empty{")" * list_len}) :- {digit_facts[:-2]}.'
        return start_list

    def get_context_facts(self, poss):
        start_list = 'start_list('
        for idx, n in enumerate(poss):
            if type(n) == torch.Tensor:
                n = int(n.item())
            start_list += f'({n}, '
        start_list += f'empty{")" * len(poss)}).'
        return start_list

    def compute_stable_models(self, asp_prog, obs, num_images, for_ilp=True):
        # Create choice rules and start list
        cr = self._create_n_choice_rules(num_images)
        cr = '\n'.join(cr)
        # Build ID and load from cache if saved
        label = int(obs.split('result(')[1].split(')')[0])
        combo_id = self._get_combo_id_num_images_label(num_images, label, for_ilp)
        if not combo_id:
            return []

        prog_id = re.sub(r"[\n\t\s]*", "", asp_prog)
        if prog_id in self.stable_model_cache and combo_id in self.stable_model_cache[prog_id]:
            return self.stable_model_cache[prog_id][combo_id]

        # Otherwise, call clingo
        start_list = self._create_start_list_choice(num_images)
        program = f'{cr}\n{start_list}\n{asp_prog}\n{obs}\n#show digit/2.'
        models = find_all_stable_models(program, self.model_to_network_preds)
        if len(models) == 0:
            print('ERROR: 0 stable models for program:')
            print(program)
            print('--------')

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
        footer = '#show result/1.'
        examples = []

        # Create examples
        if data_type in self.extra_test_nesy_data:
            loader = self.extra_test_nesy_data[data_type]
        else:
            loader = self.nesy_data[data_type]
        for ex in loader:
            # On the training set, map the NeSy image idx to the prediction idx
            if data_type == 'train' and idx_map:
                x = [lc[idx_map[i[0].item()]] for i in ex[0]]
            else:
                x = [lc[i[0].item()] for i in ex[0]]
            start_list = self.get_context_facts(x)
            examples.append(start_list)

        # Batch run evaluation
        predictions = batch_run_evaluation(header, examples, footer)
        predictions = [int(predictions[key][0].split('result(')[1].split(')')[0]) for key in
                       sorted(predictions.keys())]
        end_time = time.time() - start_time
        if data_type == 'train':
            self.logger.add_component_time(i, f'symbolic_{data_type}_{preds_type}_preds_eval', end_time)
        return predictions

    def _evaluate_mae(self, downstream_preds, loader, log=False):
        errs = []
        for idx, item in enumerate(loader):
            nsil_pred = downstream_preds[idx]
            target = item[1][0].item()
            diff = abs(target - nsil_pred)
            if log and diff >= 1:
                errs.append(math.log(diff))
            errs.append(diff)
        result_key = 'MAE'
        if log:
            result_key = 'log_MAE'
        return {result_key: np.mean(np.array(errs))}

    def manual_sum_prod_extra(self, net_out, loader):
        lc = net_out['digit']['predictions']
        preds = []
        for ex in tqdm(loader, ncols=50):
            x = [lc[i[0].item()] for i in ex[0]]
            if self.args.task_type == 'sum':
                preds.append(np.sum(np.array(x)))
            else:
                preds.append(np.prod(np.array(x)))
        return preds

    def custom_evaluation(self, i, net_out, downstream_preds):
        """
        Compute MAE on test sets
        @param i: the iteration number
        @param net_out: neural network output
        @param downstream_preds: downstream task predictions
        """
        return
        # self.logger.info('custom_eval')
        # # Firstly get results for default test loader
        # mae_results = {
        #     'default': self._evaluate_mae(downstream_preds, self.nesy_data['test'])
        # }
        #
        # # Then evaluate custom sets
        # log = False
        # if self.args.task_type == 'prod':
        #     log = True
        # for loader in self.extra_test_nesy_data:
        #     # Run fast task evaluation instead of calling clingo for large test examples
        #     _extra_preds = self.manual_sum_prod_extra(net_out, self.extra_test_nesy_data[loader])
        #     mae_results[loader] = self._evaluate_mae(_extra_preds, self.extra_test_nesy_data[loader], log=log)
        #
        # # Save to log
        # self.logger.save_custom_eval(i, mae_results)





