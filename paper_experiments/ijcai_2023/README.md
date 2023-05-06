# IJCAI 2023 Experiment Reproducability
These markdown files detail the commands required to reproduce the experimental results presented in the paper, along with Jupyter notebooks to generate the figures and tables. This file also presents details regarding the [hyper-parameters](#hyper-parameters), and [random seeds](#random-seeds) used.


## Running Commands
Connect to the container and navigate to the `scripts` directory:
```bash
docker exec -it nsil_ijcai_2023 /bin/bash
cd scripts
```
Note that when an experiment is first run, it takes a while to build the data cache. This becomes faster on subsequent runs.

## Reproducing main paper results

| Ref        | Task(s)                                                   | Links
|------------|-----------------------------------------------------------| ----------- | 
| Table 1    | Cumulative Arithmetic                                     | [Commands](./RecursiveArithmetic.md), [notebook](http://localhost:9990/notebooks/notebooks/Meta_Abd_Comparison.ipynb)
| Figure 2   | Cumulative Arithmetic neural network accuracy             | [notebook](http://localhost:9990/notebooks/notebooks/Meta_Abd_Comparison.ipynb)
| Figure 3   | Two-Digit Arithmetic                                      | [Commands](./Arithmetic.md), [notebook](http://localhost:9990/notebooks/notebooks/Epoch_vs_Accuracy_Figures.ipynb)
| Figure 4   | Hitting Set                                               | [Commands](./HittingSets.md#HS), [notebook](http://localhost:9990/notebooks/notebooks/Epoch_vs_Accuracy_Figures.ipynb)
| Section 4  | Two-Digit Arithmetic/Hitting Set: Neural network accuracy |[notebook](http://localhost:9990/notebooks/notebooks/Neural%20network%20accuracy.ipynb)
| Section 4  | Hitting Set hamming loss                                  | [notebook](http://localhost:9990/notebooks/notebooks/Hamming%20loss%20investigation%20HS%20CHS.ipynb)
| Table 2a   | Two-Digit Arithmetic E9P increasing search space          | [Commands](Arithmetic.md#increasing-the-hypothesis-space-for-E9P), [notebook](http://localhost:9990/notebooks/notebooks/MNIST_E9P_Increasing_Hypothesis_Space_Table.ipynb)
| Table 2b   | HS FashionMNIST increasing search space                   | [Commands](HittingSets.md#increasing-the-hypothesis-space-for-HS), [notebook](http://localhost:9990/notebooks/notebooks/Hitting_Sets_Increasing_Hyp_Space_Table.ipynb)
| Tables 3,4 | All ablation results                                      | [Commands](./AblationStudy.md), [notebook](http://localhost:9990/notebooks/notebooks/naive_baseline_results.ipynb)


## Reproducing supplementary material results

| Ref         | Task(s)                                          | Links
|-------------|--------------------------------------------------| ----------- | 
| Figure 6a   | Two-Digit Arithmetic E9P varying lambda          | [Commands](Arithmetic.md#varying-lambda-for-E9P), [notebook](http://localhost:9990/notebooks/notebooks/Varying_lambda_nn_weights.ipynb)
| Figure 6b,c | HS MNIST/FashionMNIST varying lambda             | [Commands](HittingSets.md#varying-lambda-for-HS), [notebook](http://localhost:9990/notebooks/notebooks/Varying_lambda_nn_weights.ipynb)
| Figure 9    | Two-Digit Arithmetic learning time               | [Commands](./Arithmetic.md), [notebook](http://localhost:9990/notebooks/notebooks/Learning-time_vs_Accuracy_Figures.ipynb)
| Figure 10   | Hitting Set learning time                        | [Commands](./HittingSets.md), [notebook](http://localhost:9990/notebooks/notebooks/Learning-time_vs_Accuracy_Figures.ipynb)


##<a name="hyper-parameters"></a> Hyper-parameters

### NSIL
The NSIL hyper-parameters used in the paper:

**Cumulative Addition:**
```json
{
  "learning_rate": 0.001030608122509068, 
  "momentum": 0.02784393448382616
}
```

**Cumulative Product:**
```json
{
  "learning_rate": 0.00010722517722295946, 
  "momentum": 0.7977277142927051
}
```

**Two-Digit Addition:**
```json
{
  "learning_rate": 0.0005404844707369889,
  "momentum": 0.446859063384064
}
```
**Two-Digit E9P:**  
```json
{
  "learning_rate": 0.0021947689485666017,
  "momentum": 0.0
}
```
**HS MNIST:**
```json
{
  "learning_rate": 0.0008253878514942516, 
  "momentum": 0.7643463721907878
}        
```
**HS FashionMNIST:**
```json
{
  "learning_rate": 0.0021947689485666017,
  "momentum": 0.0
} 
```
**CHS MNIST:**
```json
{
  "learning_rate": 0.0008253878514942516, 
  "momentum": 0.7643463721907878
}
```
**CHS FashionMNIST:**
```json
{
  "learning_rate": 0.0021947689485666017,
  "momentum": 0.0
}
```

### Baselines
The baseline hyper-parameters we obtained through tuning are available in the following files:    

| Experiment                | File 
|---------------------------| ---------- |
| Two-Digit Addition | `examples/arithmetic/baselines/results/sum/hyper-params.json`
| Two-Digit E9P   | `examples/arithmetic/baselines/results/e9p/hyper-params.json`
| Hitting Set HS            | `examples/hitting_sets/baselines/results/HS-hyper-params.json`
| Hitting Set CHS           | `examples/hitting_sets/baselines/results/CHS-hyper-params.json`

###<a name="random-seeds"></a> Random Seeds
All experiments are repeated 20 times. Each repeat uses a different random seed, generated randomly using `global_config.py`. The seeds used are:

```
7429
6670
3855
6677
5657
1591
1210
3517
2759
3531
3069
8152
5724
863
1760
5093
9456
1650
1141
1405
``` 