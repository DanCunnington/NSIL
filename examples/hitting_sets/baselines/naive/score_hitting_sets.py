from examples.hitting_sets.templates import TemplateManager
from os.path import join
import pandas as pd
import sys
import clingo

hs_rules = '''
ss(1..4).
hs_index(1..2).
elt(1..5).

:- ss_element(V1,V2); not hit(V1); elt(V2); ss(V1).
0 {hs(V1,V2) } 1 :- elt(V2); hs_index(V1).
hit(V1) :- hs(V3,V2); ss_element(V1,V2); hs_index(V3); elt(V2); ss(V1).
:- hs(V3,V1); hs(V3,V2); V1 != V2; hs_index(V3); elt(V2); elt(V1).
'''


def compute_sat(p):
    clingo_control = clingo.Control(["--warn=none", '0', '--project'])
    models = []
    try:
        clingo_control.add("base", [], p)
    except RuntimeError:
        print('Clingo runtime error')
        print('Program: {0}'.format(p))
        sys.exit(1)
    clingo_control.ground([("base", [])])

    def on_model(m):
        models.append(m)

    clingo_control.solve(on_model=on_model)
    if len(models) > 0:
        return 1
    else:
        return 0


def decide_hitting_set(x, t, templates, root_dir, args):
    train_csv = pd.read_csv(join(root_dir, f'{args.task_type}', f'{args.image_type}_train.csv'))
    _t = TemplateManager(templates, train_csv)

    # Put digit predictions into template
    facts = _t.templates[t].get_ss_element_facts_with_digits(x)
    return compute_sat(f'{hs_rules}\n{facts}')