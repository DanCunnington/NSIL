from src.NSILTask.base import NSILTask
from src.ILP.WCDPI import WCDPI
from src.ASPUtils import find_all_stable_models, batch_run_evaluation
import time
import torch
import re


class ArithmeticTask(NSILTask):
    def __init__(self, data, runner, net_confs, image_to_net_map, digit_vals, ilp_config=None, n_digits=2):
        """
        Arithmetic Tasks
        @param data: NSILDataset
        @param runner: NSILRun instance
        @param net_confs: dict of NSIL neural network configurations
        @param image_to_net_map: image to network map
        @param digit_vals: possible digit values (e.g. [0,1,...,9])
        @param ilp_config: custom bk and md
        @param n_digits: number of digits in each example
        """
        self.digit_vals = digit_vals
        self.n_digits = n_digits
        if ilp_config:
            bk = ilp_config.bk
            md = ilp_config.md
        else:
            bk = '''
            even(X) :- digit_type(X), X \\ 2 = 0.
            plus_nine(X1,X2) :- digit_type(X1), X2=9+X1.
            
            :- digit(1,X0), digit(2,X1), result(Y1), result(Y2), Y1 != Y2.
            result(Y) :- digit(1,X0), digit(2,X1), solution(X0,X1,Y).
            num(0..18).
            digit_type(0..9).
            '''

            md = f'''
            #modeh(solution(var(digit_type),var(digit_type),var(num))).
            #modeb(var(num) = var(digit_type)).
            #modeb(var(num) = var(digit_type) + var(digit_type)).
            #modeb(plus_nine(var(digit_type),var(num))).
            #modeb(even(var(digit_type))).
            #modeb(not even(var(digit_type))).
            #maxv(3).
            
            #bias("penalty(1, head(X)) :- in_head(X).").
            #bias("penalty(1, body(X)) :- in_body(X).").
            '''

        # Remove spaces introduced by multi-line string
        def clean(x): return '\n'.join([l.strip() for l in x.split('\n')])

        md = clean(md)
        bk = clean(bk)
        super().__init__(data, runner, net_confs, image_to_net_map, md, bk)

    def _create_n_choice_rules(self, n):
        """
        Create choice rules for each digit fact
        @param n: number of digits
        @return: choice rules
        """
        cr = []
        for d in range(n):
            _d_rule = '1 {'
            _d_rule += '; '.join([f'digit({d + 1},{v})' for v in self.digit_vals])
            _d_rule = f'{_d_rule} }} 1.'
            cr.append(_d_rule)
        return cr

    def create_bootstrap_task(self):
        # For each unique_label in the dataset, create LAS example with choice rules for possible digit values
        ex_id = 0
        poss_labels = self.nesy_data['train'].dataset.unique_labels
        for pl in poss_labels:
            self.bootstrap_examples[f'bootstrap_{ex_id}'] = WCDPI(
                ex_id=f'bootstrap_{ex_id}',
                positive=True,
                weight='inf',
                inclusion=[f'result({pl})'],
                exclusion=[],
                context=self._create_n_choice_rules(self.n_digits)
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
        converted_h = h
        self.logger.info('converted_h', converted_h)
        return converted_h

    def model_to_network_preds(self, m):
        m = sorted(m.split(' '))
        # Extract network predictions from stable model
        net_preds = [int(f.split(f'digit({idx + 1},')[1].split(')')[0]) for idx, f in enumerate(m)]
        return net_preds

    def get_combo_id(self, ex, for_ilp=True):
        label = ex[1][0].item()
        return f'label_{label}'

    def get_context_facts(self, poss):
        facts = ''
        for idx, n in enumerate(poss):
            if type(n) == torch.Tensor:
                n = int(n.item())
            facts += f'digit({idx+1},{int(n)}). '
        facts = facts[:-1]
        return facts

    def compute_stable_models(self, asp_prog, obs, num_images, for_ilp=True):
        # Create choice rules
        cr = self._create_n_choice_rules(self.n_digits)
        cr = '\n'.join(cr)
        program = f'{cr}\n{asp_prog}\n{obs}\n#show digit/2.'

        # Build ID and load from cache if saved
        label = obs.split('result(')[1].split(')')[0]
        combo_id = f'label_{label}'
        prog_id = re.sub(r"[\n\t\s]*", "", asp_prog)
        if prog_id in self.stable_model_cache and combo_id in self.stable_model_cache[prog_id]:
            return self.stable_model_cache[prog_id][combo_id]

        # Otherwise, call clingo
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
        for ex in self.nesy_data[data_type]:
            # On the training set, map the NeSy image idx to the prediction idx
            if data_type == 'train' and idx_map:
                x = [lc[idx_map[i[0].item()]] for i in ex[0]]
            else:
                x = [lc[i[0].item()] for i in ex[0]]
            facts = self.get_context_facts(x)
            examples.append(facts)

        # Batch run evaluation
        predictions = batch_run_evaluation(header, examples, footer)

        predictions = [int(predictions[key][0].split('result(')[1].split(')')[0]) for key in
                       sorted(predictions.keys())]
        end_time = time.time() - start_time
        if data_type == 'train':
            self.logger.add_component_time(i, f'symbolic_{data_type}_{preds_type}_preds_eval', end_time)
        return predictions

    def custom_evaluation(self, i, net_out, downstream_preds):
        """
        Compute MAE on test sets
        @param i: the iteration number
        @param h: the current hypothesis
        @param net_out: neural network output
        @param downstream_preds: downstream task predictions
        """
        return





