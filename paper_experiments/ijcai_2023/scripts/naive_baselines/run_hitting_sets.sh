# Ensure experiment is passed and run setup
while getopts d:m: flag
do
    case "${flag}" in
        d) DATASET=${OPTARG};;
        m) METHOD=${OPTARG};;
    esac
done
if [ -z $DATASET ]; then DATASET=mnist; fi
if [ -z $METHOD ]; then METHOD=ff_nsl; fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

EXPERIMENT="hitting_sets/baselines/naive/${METHOD}"
source $SCRIPT_DIR/setup.sh ${EXPERIMENT}
echo EXPERIMENT: ${EXPERIMENT}
echo DATASET: ${DATASET}

# Run task
python -u train.py --image_type ${DATASET} --task_type=hs

# For NeurASP, need to train with chs also
if [[ $METHOD == *"NeurASP"* ]]; then
  python -u train.py --image_type ${DATASET} --task_type=chs
fi
cd ../
python -u evaluate.py --method ${METHOD} --image_type ${DATASET} --task_type=hs
python -u evaluate.py --method ${METHOD} --image_type ${DATASET} --task_type=chs
