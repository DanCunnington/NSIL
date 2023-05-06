# Ensure experiment is passed and run setup
while getopts p:s:m:a: flag
do
    case "${flag}" in
        p) DATA_PCT=${OPTARG};;
        s) START_SEED_IDX=${OPTARG};;
        m) MAX_SEED_NUM=${OPTARG};;
        a) ADDITIONAL_ARGS=${OPTARG};;
    esac
done
if [ -z $DATA_PCT ]; then DATA_PCT=100; fi
if [ -z $START_SEED_IDX ]; then START_SEED_IDX=0; fi
if [ -z $MAX_SEED_NUM ]; then MAX_SEED_NUM=20; fi
if [ -z "$ADDITIONAL_ARGS" ]; then
    ADDITIONAL_ARGS=""
else
    NEW_ARGS=""
    SPLITTED_ADD_ARGS=$(echo $ADDITIONAL_ARGS | tr " " "\n")
    for arg in $SPLITTED_ADD_ARGS
    do
      NEW_ARGS="${NEW_ARGS} --${arg} "
    done
    ADDITIONAL_ARGS=$NEW_ARGS
fi

if [[ "$ADDITIONAL_ARGS" == *"task_type=chs"* ]]; then
  SUB_RESULT_DIR="CHS"
else
  SUB_RESULT_DIR="HS"
fi

if [[ "$ADDITIONAL_ARGS" == *"image_type=fashion_mnist"* ]]; then
  SUB_RESULT_DIR="${SUB_RESULT_DIR}_fashion_mnist"
elif [[ "$ADDITIONAL_ARGS" == *"image_type=cifar_10"* ]]; then
  SUB_RESULT_DIR="${SUB_RESULT_DIR}_cifar_10"
else
  SUB_RESULT_DIR="${SUB_RESULT_DIR}_mnist"
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

EXPERIMENT=hitting_sets
source $SCRIPT_DIR/setup.sh ${EXPERIMENT}
echo EXPERIMENT: ${EXPERIMENT}
echo SUB_RESULT_DIR: ${SUB_RESULT_DIR}
echo START_SEED_IDX: ${START_SEED_IDX}
echo MAX_SEED_NUM: ${MAX_SEED_NUM}
echo ADDITIONAL_ARGS: ${ADDITIONAL_ARGS}

# Run task
IFS=$'\n' read -d '' -r -a seeds < $BASE_PATH/seeds.txt
COUNT=0
for idx in "${!seeds[@]}"
do
  if [ $COUNT -lt $MAX_SEED_NUM ]
  then
    SEED_IDX=$((START_SEED_IDX+idx))
    REPEAT_ID=$((START_SEED_IDX+idx+1))
    RESULTS_DIR="results/no_corrective_ex_repeats/${SUB_RESULT_DIR}/${DATA_PCT}/${REPEAT_ID}"

    mkdir -p $RESULTS_DIR
    python run_with_given_h.py --ilp_system=ILASP --pylasp --pct ${DATA_PCT} --skip_symbolic_learning --skip_corrective_examples --save_dir $RESULTS_DIR --seed ${seeds[SEED_IDX]} --save_nets ${ADDITIONAL_ARGS}
    COUNT=$((COUNT+1))
  fi
done