# Ensure experiment is passed and run setup
while getopts m: flag
do
    case "${flag}" in
        m) METHOD=${OPTARG};;
    esac
done
if [ -z $METHOD ]; then METHOD=ff_nsl; fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

EXPERIMENT="recursive_arithmetic/baselines/naive/${METHOD}"
source $SCRIPT_DIR/setup.sh ${EXPERIMENT}
echo EXPERIMENT: ${EXPERIMENT}

# Run task
python -u train.py --task_type=sum
python -u train.py --task_type=prod
cd ../
python -u evaluate.py --method ${METHOD} --task_type=sum
python -u evaluate.py --method ${METHOD} --task_type=prod
