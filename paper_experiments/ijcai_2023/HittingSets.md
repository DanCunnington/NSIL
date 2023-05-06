# Hitting Set Experiments

### HS
```bash
./run_hitting_sets_repeats.sh -s 0 -m 5
./run_hitting_sets_repeats.sh -s 5 -m 5
./run_hitting_sets_repeats.sh -s 10 -m 5
./run_hitting_sets_repeats.sh -s 15 -m 5

./run_hitting_sets_repeats.sh -s 0 -m 5 -a "image_type=fashion_mnist"
./run_hitting_sets_repeats.sh -s 5 -m 5 -a "image_type=fashion_mnist"
./run_hitting_sets_repeats.sh -s 10 -m 5 -a "image_type=fashion_mnist"
./run_hitting_sets_repeats.sh -s 15 -m 5 -a "image_type=fashion_mnist"

./run_hitting_sets_repeats.sh -s 0 -m 5 -a "image_type=cifar_10"
./run_hitting_sets_repeats.sh -s 5 -m 5 -a "image_type=cifar_10"
./run_hitting_sets_repeats.sh -s 10 -m 5 -a "image_type=cifar_10"
./run_hitting_sets_repeats.sh -s 15 -m 5 -a "image_type=cifar_10"
```

### CHS
```bash
./run_hitting_sets_repeats.sh -s 0 -m 5 -a "task_type=chs"
./run_hitting_sets_repeats.sh -s 5 -m 5 -a "task_type=chs"
./run_hitting_sets_repeats.sh -s 10 -m 5 -a "task_type=chs"
./run_hitting_sets_repeats.sh -s 15 -m 5 -a "task_type=chs"

./run_hitting_sets_repeats.sh -s 0 -m 5 -a "task_type=chs image_type=fashion_mnist"
./run_hitting_sets_repeats.sh -s 5 -m 5 -a "task_type=chs image_type=fashion_mnist"
./run_hitting_sets_repeats.sh -s 10 -m 5 -a "task_type=chs image_type=fashion_mnist"
./run_hitting_sets_repeats.sh -s 15 -m 5 -a "task_type=chs image_type=fashion_mnist"

./run_hitting_sets_repeats.sh -s 0 -m 5 -a "task_type=chs image_type=cifar_10"
./run_hitting_sets_repeats.sh -s 5 -m 5 -a "task_type=chs image_type=cifar_10"
./run_hitting_sets_repeats.sh -s 10 -m 5 -a "task_type=chs image_type=cifar_10"
./run_hitting_sets_repeats.sh -s 15 -m 5 -a "task_type=chs image_type=cifar_10"
```

### Increasing the hypothesis space for HS
```bash
./run_HS_increasing_hyp_space.sh --ilp_config=config_1
./run_HS_increasing_hyp_space.sh --ilp_config=config_2
./run_HS_increasing_hyp_space.sh --ilp_config=config_3
./run_HS_increasing_hyp_space.sh --ilp_config=config_4
./run_HS_increasing_hyp_space.sh --ilp_config=config_5
./run_HS_increasing_hyp_space.sh --ilp_config=config_6
./run_HS_increasing_hyp_space.sh --ilp_config=config_7
./run_HS_increasing_hyp_space.sh --ilp_config=config_8
./run_HS_increasing_hyp_space.sh --ilp_config=config_9
./run_HS_increasing_hyp_space.sh --ilp_config=config_10
```


### Varying lambda for HS
```bash
./run_HS_varying_lambda.sh --l=1
./run_HS_varying_lambda.sh --l=0.8
./run_HS_varying_lambda.sh --l=0.6
./run_HS_varying_lambda.sh --l=0.4
./run_HS_varying_lambda.sh --l=0.2
./run_HS_varying_lambda.sh --l=0

./run_HS_varying_lambda.sh --l=1 --image_type=fashion_mnist
./run_HS_varying_lambda.sh --l=0.8 --image_type=fashion_mnist
./run_HS_varying_lambda.sh --l=0.6 --image_type=fashion_mnist
./run_HS_varying_lambda.sh --l=0.4 --image_type=fashion_mnist
./run_HS_varying_lambda.sh --l=0.2 --image_type=fashion_mnist
./run_HS_varying_lambda.sh --l=0 --image_type=fashion_mnist
```