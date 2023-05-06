import clingo
import sys
import numpy as np
import multiprocessing
from global_config import NUM_CPU


def find_all_stable_models(p, model_convert_fn):
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
        models.append(model_convert_fn(str(m)))

    clingo_control.solve(on_model=on_model)
    return models


def run_clingo_parallel(tasks, rtn_keys, rtn_dict):
    current_key = None
    current_models = None

    def on_finish(r):
        # Add predictions to the results dictionary
        if current_key is not None:
            rtn_dict[int(rtn_keys[current_key])] = current_models
        else:
            print('current_key is none')
            sys.exit(1)

    def on_model(m):
        current_models.append(str(m))

    for task_idx, task in enumerate(tasks):
        current_key = task_idx
        current_models = []
        ctl = clingo.Control(["0"])
        ctl.add("base", [], task)
        ctl.ground([("base", [])])
        ctl.solve(on_finish=on_finish, on_model=on_model)


def batch_run_evaluation(header, examples, footer):
    tasks = []
    ex_ids = []

    for ex_idx, ex in enumerate(examples):
        # Build task
        task = header + '\n' + ex + '\n' + footer
        ex_ids.append(ex_idx)
        tasks.append(task)

    task_splits = np.array_split(tasks, NUM_CPU)
    ex_id_splits = np.array_split(ex_ids, NUM_CPU)
    manager = multiprocessing.Manager()
    results = manager.dict()
    jobs = []

    for idx, split in enumerate(task_splits):
        p = multiprocessing.Process(target=run_clingo_parallel,
                                    args=(split,
                                          ex_id_splits[idx],
                                          results))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    return results
