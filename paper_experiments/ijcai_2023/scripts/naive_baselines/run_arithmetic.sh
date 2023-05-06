# Ensure experiment is passed and run setup
while getopts p:m: flag
do
    case "${flag}" in
        p) DATA_PCT=${OPTARG};;
        m) METHOD=${OPTARG};;
    esac
done
if [ -z $DATA_PCT ]; then DATA_PCT=100; fi
if [ -z $METHOD ]; then METHOD=ff_nsl; fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

EXPERIMENT="arithmetic/baselines/naive/${METHOD}"
source $SCRIPT_DIR/setup.sh ${EXPERIMENT}
echo EXPERIMENT: ${EXPERIMENT}
echo DATA_PCT: ${DATA_PCT}

# Run task
python -u train.py --pct ${DATA_PCT} --task_type=sum

# For NeurASP, need to train with e9p also
if [[ $METHOD == *"NeurASP"* ]]; then
  python -u train.py --pct ${DATA_PCT} --task_type=e9p
fi
cd ../
python -u evaluate.py --method ${METHOD} --pct ${DATA_PCT} --task_type=sum
python -u evaluate.py --method ${METHOD} --pct ${DATA_PCT} --task_type=e9p
