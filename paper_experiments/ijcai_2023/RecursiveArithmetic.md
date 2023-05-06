# Cumulative Arithmetic Experiments
Run NSIL tasks for 20 repeats with different seeds. The script enables parallel runs of different seeds via the `-s` (start index) and `-m` (max number) flags. The `-p` flag changes the dataset percentages.


Single runs:
```bash
cd ../../examples/recursive_arithmetic
python run.py --ilp_system=ILASP --custom_ilp_cmd_line_args='--version=4' --meta_abd_data --num_iterations=50 --prune_ilp_example_weight_threshold=5
python run.py --ilp_system=ILASP --custom_ilp_cmd_line_args='--version=4' --meta_abd_data --num_iterations=50 --prune_ilp_example_weight_threshold=5 --task_type=prod
```



### Addition

```bash
./run_recursive_arithmetic_repeats.sh -s 0 -m 5
./run_recursive_arithmetic_repeats.sh -s 5 -m 5
./run_recursive_arithmetic_repeats.sh -s 10 -m 5 
./run_recursive_arithmetic_repeats.sh -s 15 -m 5
```

### Product
```bash
./run_recursive_arithmetic_repeats.sh -s 0 -m 5 -a "task_type=prod"
./run_recursive_arithmetic_repeats.sh -s 5 -m 5 -a "task_type=prod"
./run_recursive_arithmetic_repeats.sh -s 10 -m 5 -a "task_type=prod"
./run_recursive_arithmetic_repeats.sh -s 15 -m 5 -a "task_type=prod"
```