SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

EXPERIMENT=arithmetic
source $SCRIPT_DIR/setup.sh ${EXPERIMENT}

# Run task
PCT=40
ARGS="$@"
eval "arr=($ARGS)"
# Get config
ILP_CONF=""
ILP_EX_W=1
for s in "${arr[@]}"; do
    if [[ "$s" == *"ilp_config="* ]]; then
      ILP_CONF=${s#*=}
    fi
done


if [[ "$ILP_CONF" == "" ]]; then
  echo "Must specify ILP config"
  exit
fi

# Prune weights in large search spaces
if [[ "$ILP_CONF" == "config_4" ] || [ "$ILP_CONF" == "config_5" ] || [ "$ILP_CONF" == "config_6" ]]; then
  ILP_EX_W=5
fi

RESULTS_DIR="results/increasing_hyp_space/e9p/${PCT}/${ILP_CONF}"
mkdir -p $RESULTS_DIR
python run.py --pct ${PCT} --save_dir $RESULTS_DIR --num_iterations 5 --task_type=e9p --prune_ilp_example_weight_threshold=${ILP_EX_W} --ilp_config=${ILP_CONF}

