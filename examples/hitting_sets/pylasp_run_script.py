#ilasp_script
import json
import sys
ilasp.cdilp.initialise()
ilasp.cdilp.set_clingo_args("");
solve_result = ilasp.cdilp.solve()

c_egs = ilasp.find_all_counterexamples(solve_result)

conflict_analysis_strategy = {
  'positive-strategy': 'all-ufs',
  'negative-strategy': 'single-as',
  'brave-strategy':    'all-ufs',
  'cautious-strategy': 'single-as-pair'
}

# Weights specified by NSIL
weights = <<NSIL_WEIGHTS_TO_REPLACE>>
while c_egs and solve_result is not None:
  while c_egs:
    # Sort examples by descending weight
    c_eg_weights = {}
    for ce in c_egs:
      w = weights[ce['id']]

      # Set infinite weights to a really high number for sorting
      if w == 'inf':
        w = 999999999999
      else:
        w = int(w)
      c_eg_weights[ce['id']] = w

    sorted_ce_eg_weights = sorted(c_eg_weights.items(), key=lambda x: x[1], reverse=True)

    # Take counter example with the maximum weight
    ce = ilasp.get_example(sorted_ce_eg_weights[0][0])

    constraint = ilasp.cdilp.analyse_conflict(solve_result['hypothesis'], ce['id'], conflict_analysis_strategy)

    c_eg_ids = list(map(lambda x: x['id'], c_egs))

    prop_egs = []
    if weights[ce['id']] == 'inf':
      c_egs = []
      prop_egs = [ce['id']]
    else:
      if ce['type'] == 'positive':
        prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_eg_ids, {'select-examples': ['positive'], 'strategy': 'cdpi-implies-constraint'})
      elif ce['type'] == 'negative':
        prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_eg_ids, {'select-examples': ['negative'], 'strategy': 'neg-constraint-implies-cdpi'})
      elif ce['type'] == 'brave-order':
        prop_egs = ilasp.cdilp.propagate_constraint(constraint, c_eg_ids, {'select-examples': ['brave-order'],    'strategy': 'cdoe-implies-constraint'})
      else:
        prop_egs = [ce['id']]

      c_egs = list(map(lambda id: ilasp.get_example(id), list(set(c_eg_ids) - set(prop_egs))))

    ilasp.cdilp.add_coverage_constraint(constraint, prop_egs)

  solve_result = ilasp.cdilp.solve()

  if solve_result is not None:
    c_egs = ilasp.find_all_counterexamples(solve_result)


if solve_result:
  print(ilasp.hypothesis_to_string(solve_result['hypothesis']))
  sys.stdout.flush()
else:
  print('UNSATISFIABLE')
  sys.stdout.flush()

sys.stdout.flush()
#end.
