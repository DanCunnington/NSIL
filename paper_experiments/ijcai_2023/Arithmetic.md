# Two-Digit Arithmetic Experiments

### Addition
```
./run_arithmetic_repeats.sh -p 100 -s 0 -m 5 
./run_arithmetic_repeats.sh -p 100 -s 5 -m 5 
./run_arithmetic_repeats.sh -p 100 -s 10 -m 5 
./run_arithmetic_repeats.sh -p 100 -s 15 -m 5 

./run_arithmetic_repeats.sh -p 10 -s 0 -m 10 
./run_arithmetic_repeats.sh -p 10 -s 10 -m 10 

./run_arithmetic_repeats.sh -p 5 -s 0 -m 10 
./run_arithmetic_repeats.sh -p 5 -s 10 -m 10 

./run_arithmetic_repeats.sh -p 1 -s 0 -m 20 
```

### E9P
```
./run_arithmetic_repeats.sh -p 100 -s 0 -m 5 -a "task_type=e9p"
./run_arithmetic_repeats.sh -p 100 -s 5 -m 5 -a "task_type=e9p"
./run_arithmetic_repeats.sh -p 100 -s 10 -m 5 -a "task_type=e9p"
./run_arithmetic_repeats.sh -p 100 -s 15 -m 5 -a "task_type=e9p"

./run_arithmetic_repeats.sh -p 10 -s 0 -m 10 -a "task_type=e9p"
./run_arithmetic_repeats.sh -p 10 -s 10 -m 10 -a "task_type=e9p"

./run_arithmetic_repeats.sh -p 5 -s 0 -m 10 -a "task_type=e9p"
./run_arithmetic_repeats.sh -p 5 -s 10 -m 10 -a "task_type=e9p"

./run_arithmetic_repeats.sh -p 1 -s 0 -m 20 -a "task_type=e9p"
```


### Increasing the hypothesis space for E9P
```bash
./run_e9p_increasing_hyp_space.sh --ilp_config=config_1
./run_e9p_increasing_hyp_space.sh --ilp_config=config_2
./run_e9p_increasing_hyp_space.sh --ilp_config=config_3
./run_e9p_increasing_hyp_space.sh --ilp_config=config_4
./run_e9p_increasing_hyp_space.sh --ilp_config=config_5
./run_e9p_increasing_hyp_space.sh --ilp_config=config_6
./run_e9p_increasing_hyp_space.sh --ilp_config=config_7

```


### Varying lambda for E9P
```bash
./run_e9p_varying_lambda.sh --l=1
./run_e9p_varying_lambda.sh --l=0.8
./run_e9p_varying_lambda.sh --l=0.6
./run_e9p_varying_lambda.sh --l=0.4
./run_e9p_varying_lambda.sh --l=0.2
./run_e9p_varying_lambda.sh --l=0
```