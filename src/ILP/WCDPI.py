
class WCDPI:
    def __init__(self, ex_id, positive, weight, inclusion, exclusion, context,
                 group=None):
        """
        Base WCDPI class
        @param ex_id: the example ID (str)
        @param positive: positive or negative WCDPI (bool)
        @param weight: the example weight (int)
        @param inclusion: inclusion atoms (str)
        @param exclusion: exclusion atoms (str)
        @param context: contextual atoms (str)
        @param group: group assignment if using meta-level injection
        """
        self.ex_id = ex_id
        self.positive = positive
        self.weight = weight
        self.inclusion = inclusion
        self.exclusion = exclusion
        self.context = context
        self.group = group

    def __str__(self):
        """
        Return a string representation of a WCDPI object
        :return: WCDPI string
        :rtype: str
        """
        if self.positive:
            pos_or_neg = 'pos'
        else:
            pos_or_neg = 'neg'
        if self.weight == 'inf' or self.group:
            wcdpi_str = '#{0}({1}, '.format(pos_or_neg, self.ex_id)
        else:
            wcdpi_str = '#{0}({1}@{2}, '.format(pos_or_neg, self.ex_id, self.weight)

        inclusion_str = ', '.join(self.inclusion)
        exclusion_str = ', '.join(self.exclusion)
        context_str = '\n\t'.join(self.context)

        wcdpi_str += '{{ {0} }}, '.format(inclusion_str)
        wcdpi_str += '{{ {0} }}, {{\n'.format(exclusion_str)
        wcdpi_str += '\t{0}'.format(context_str)
        wcdpi_str += '\n}).'

        # Add meta-level injection groupings and weight
        if self.group:
            # Group assignment
            ga = '#inject("group(group_{0},{1}).").'.format(self.group, self.ex_id)

            # Weight inject rule
            if self.weight == 'inf':
                ir = '#inject("example_active({0}).").'.format(self.ex_id)
            else:
                ir = '#inject("weight({0},{1}).").'.format(self.ex_id, self.weight)

            wcdpi_str += '\n{0}\n{1}'.format(ga, ir)

        return wcdpi_str


class DefaultWCDPIPair:
    def __init__(self, combo_id, label, ctx_facts):
        self.start_weight = 0
        explore_ctx = [f'result :- result(X), X != {label}.',
                       ':- result(X), result(Y), Y < X.']
        explore_ctx += ctx_facts

        exploit_ctx = [f':- result(X), X != {label}.']
        exploit_ctx += ctx_facts

        self.explore = WCDPI(ex_id=f'{combo_id}_explore',
                             positive=True,
                             weight=self.start_weight,
                             inclusion=['result'],
                             exclusion=[],
                             context=explore_ctx)
        self.exploit = WCDPI(ex_id=f'{combo_id}_exploit',
                             positive=True,
                             weight=self.start_weight,
                             inclusion=[f'result({label})'],
                             exclusion=[],
                             context=exploit_ctx)

    def __str__(self):
        ex_1_str = str(self.explore)
        ex_2_str = str(self.exploit)
        return f'{ex_1_str}\n{ex_2_str}'


class MetaLevelInjectionBinaryWCDPIPair:
    def __init__(self, combo_id, groups, ctx_facts):
        start_weight = 1
        self.pos = WCDPI(f'{combo_id}_pos',
                         positive=True,
                         weight=start_weight,
                         inclusion=[],
                         exclusion=[],
                         context=ctx_facts,
                         group=groups[0])
        self.neg = WCDPI(f'{combo_id}_neg',
                         positive=False,
                         weight=start_weight,
                         inclusion=[],
                         exclusion=[],
                         context=ctx_facts,
                         group=groups[1])

    def __str__(self):
        ex_1_str = str(self.pos)
        ex_2_str = str(self.neg)
        return f'{ex_1_str}\n{ex_2_str}'

    def get_example_ids_and_weights(self):
        return [
            (self.pos.ex_id, self.pos.weight),
            (self.neg.ex_id, self.neg.weight)
        ]