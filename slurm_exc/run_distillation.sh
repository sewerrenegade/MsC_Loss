#!/bin/bash
#SBATCH --job-name=kd-exp                                           # Job name
#SBATCH --output=/home/icb/yufan.xia/milad.bassil/logs/wandb_agent_%A_%a.out # Output file
#SBATCH --error=/home/icb/yufan.xia/milad.bassil/logs/wandb_agent_%A_%a.out  # Error file
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1

CONFIG_NAME=$1

if [ -z "$CONFIG_NAME" ]; then
    echo "Error: Config name not provided."
    echo "Usage: sbatch run_exp.sh <config_name>"
    exit 1
fi

echo "Running config: $CONFIG_NAME"

cd /home/icb/yufan.xia/milad.bassil/Topological_Knowledge_Distillation
source /home/icb/yufan.xia/tools/apps/mamba/etc/profile.d/conda.sh
conda init
conda activate milad
echo "Activated environment: $CONDA_DEFAULT_ENV"

wandb login

python train.py --config-path configs/hydra --config-name "$CONFIG_NAME"
