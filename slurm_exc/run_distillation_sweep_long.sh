#!/bin/bash
#SBATCH --job-name=kd-exp                                           # Job name
#SBATCH --output=/home/icb/yufan.xia/milad.bassil/logs/wandb_agent_%A_%a.out # Output file
#SBATCH --error=/home/icb/yufan.xia/milad.bassil/logs/wandb_agent_%A_%a.out  # Error file
#SBATCH --time=44:00:00
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_long
#SBATCH --nodes=1


cd /home/icb/yufan.xia/milad.bassil/Topological_Knowledge_Distillation
source /home/icb/yufan.xia/tools/apps/mamba/etc/profile.d/conda.sh
conda init
conda activate milad
echo "Activated environment: $CONDA_DEFAULT_ENV"
wandb login

#wandb agent milad-research/final_distillaiton_2/0dfmjq9o ## aceve
wandb agent milad-research/final_distillaiton_2/lgewqbra ##bm imangeA