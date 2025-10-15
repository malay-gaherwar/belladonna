#!/usr/bin/env bash
#SBATCH --job-name=belladonna-demo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# Activate conda env
source ~/.bashrc
conda activate belladonna

echo "Running sample task on $(hostname)"
python -m belladonna --print-config
