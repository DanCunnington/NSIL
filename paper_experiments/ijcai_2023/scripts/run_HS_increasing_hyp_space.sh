SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

EXPERIMENT=hitting_sets
source $SCRIPT_DIR/setup.sh ${EXPERIMENT}

# Run task
PCT=100
ARGS="$@"
eval "arr=($ARGS)"
# Get config
ILP_CONF=""
for s in "${arr[@]}"; do
    if [[ "$s" == *"ilp_config="* ]]; then
      ILP_CONF=${s#*=}
    fi
done


if [[ "$ILP_CONF" == "" ]]; then
  echo "Must specify ILP config"
  exit
fi

RESULTS_DIR="results/increasing_hyp_space/HS_fashion_mnist/${PCT}/${ILP_CONF}"
mkdir -p $RESULTS_DIR
python run.py --ilp_system=ILASP --pylasp --pct ${PCT} --save_dir ${RESULTS_DIR} --num_iterations 5 --image_type=fashion_mnist --ilp_config=${ILP_CONF}

