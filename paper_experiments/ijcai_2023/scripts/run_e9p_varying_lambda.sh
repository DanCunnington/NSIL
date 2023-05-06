SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

EXPERIMENT=arithmetic
source $SCRIPT_DIR/setup.sh ${EXPERIMENT}

# Run task
PCT=40
ARGS="$@"
eval "arr=($ARGS)"
# Get config
LAMBDA=""
START_REPEAT=0
NUM_REPEATS=5
for s in "${arr[@]}"; do
    if [[ "$s" == *"l="* ]]; then
      LAMBDA=${s#*=}
    fi
    if [[ "$s" == *"s="* ]]; then
      START_REPEAT=${s#*=}
    fi
    if [[ "$s" == *"m="* ]]; then
      NUM_REPEATS=${s#*=}
    fi
done


if [[ "$LAMBDA" == "" ]]; then
  echo "Must specify lambda value"
  exit
fi

IFS=$'\n' read -d '' -r -a seeds < $BASE_PATH/seeds.txt
COUNT=0
for idx in "${!seeds[@]}"
do
  if [ $COUNT -lt $NUM_REPEATS ]
  then
    SEED_IDX=$((START_REPEAT+idx))
    REPEAT_ID=$((START_REPEAT+idx+1))
    RESULTS_DIR="results/varying_lambda/e9p/${PCT}/${LAMBDA}/${REPEAT_ID}"
    mkdir -p $RESULTS_DIR
    python run.py --pct ${PCT} --save_dir $RESULTS_DIR --seed ${seeds[SEED_IDX]} --num_iterations 10 --task_type=e9p --exploit_ex_lr=${LAMBDA}
    COUNT=$((COUNT+1))
  fi
done
