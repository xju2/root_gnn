#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --account=m3443
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --time=5:00:00
#SBATCH --job-name=train_cfc_1
#SBATCH --mail-user=arihant_kaul@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH --qos=regular

# edit this line to change which model is being used
if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

wandb agent ${SWEEP_ID}

SCRIPT_EXIT_CODE=$?

if [ $SCRIPT_EXIT_CODE -ne 0 ]; then
    echo "Script failed with exit code $SCRIPT_EXIT_CODE"
    exit 1
fi


exit $SCRIPT_EXIT_CODE