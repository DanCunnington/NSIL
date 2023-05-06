import torch


class TemplateManager:
    def __init__(self, all_set_configs, train_csv):
        """
        Store template and group information
        @param all_set_configs: possible element set configurations
        @param train_csv: csv file for the training set
        """
        self.templates = [Template(idx, c) for idx, c in enumerate(all_set_configs)]
        self.train_csv = train_csv

        # Create groups
        unique_combos = self.train_csv.groupby(['template', 'label'], as_index=False).first().values
        self.groups_by_template = {}
        self.labels_by_template = {}
        group_id = 1
        for item in unique_combos:
            key = item[0]
            val = item[1]
            if key in self.groups_by_template:
                self.groups_by_template[key].append(group_id)
                self.labels_by_template[key].append(val)
            else:
                self.groups_by_template[key] = [group_id]
                self.labels_by_template[key] = [val]
            group_id += 1

    def get_group_ids_for_template(self, t):
        return self.groups_by_template[t.t_id]

    def get_labels_for_template(self, t):
        return self.labels_by_template[t.t_id]

    def get_template_by_id(self, t_id):
        return self.templates[t_id]


class Template:
    def __init__(self, t_id, set_config):
        self.t_id = t_id
        self.set_config = set_config

    def get_constraints(self):
        # Generate ss_element_rules
        rules = []
        constrs = ['% Make sure digits in the same set can\'t have the same value']
        digit_id = 1
        for idx, digits in enumerate(self.set_config):
            ss_id = idx + 1
            for _ in digits:
                rules.append(f'ss_element({ss_id},X) :- digit({digit_id},X).')
                digit_id += 1

            # Generate set constraints
            if len(digits) == 2:
                constrs.append(f':- digit({digit_id - 2},X), digit({digit_id - 1},X).')
            elif len(digits) == 3:
                constrs.append(
                    f':- digit({digit_id - 3},X), digit({digit_id - 2},X), digit({digit_id - 1},X).')
                constrs.append(f':- digit({digit_id - 3},X), digit({digit_id - 2},X).')
                constrs.append(f':- digit({digit_id - 3},X), digit({digit_id - 1},X).')
                constrs.append(f':- digit({digit_id - 2},X), digit({digit_id - 1},X).')
        prog = rules + constrs
        return prog

    def get_ss_element_facts_with_digits(self, digits, increment=True):
        facts = ''
        dig_id = 0
        for idx, subset in enumerate(self.set_config):
            fact = f'ss_element({idx + 1},'
            for _ in subset:
                d = digits[dig_id]
                if type(d) == torch.Tensor:
                    d = d.item()
                d = int(d)
                if increment:
                    d = d + 1
                facts += f'{fact}{d}).\n\t'
                dig_id += 1
        return facts
