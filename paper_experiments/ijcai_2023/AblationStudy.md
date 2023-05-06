# Ablation Study Experiments


## Arithmetic
```bash
naive_baselines/run_arithmetic.sh -m ff_nsl -p 100
naive_baselines/run_arithmetic.sh -m ff_nsl -p 10
naive_baselines/run_arithmetic.sh -m ff_nsl -p 5

naive_baselines/run_arithmetic.sh -m NeurASP -p 100
naive_baselines/run_arithmetic.sh -m NeurASP -p 10
naive_baselines/run_arithmetic.sh -m NeurASP -p 5
```

## Recursive Arithmetic
```bash
naive_baselines/run_recursive_arithmetic.sh -m ff_nsl
naive_baselines/run_recursive_arithmetic.sh -m NeurASP
```

## Hitting Sets
```bash
naive_baselines/run_hitting_sets.sh -m ff_nsl -d mnist
naive_baselines/run_hitting_sets.sh -m ff_nsl -d fashion_mnist
naive_baselines/run_hitting_sets.sh -m NeurASP -d mnist
naive_baselines/run_hitting_sets.sh -m NeurASP -d fashion_mnist
```