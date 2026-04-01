#!/bin/bash
#SBATCH --job-name=EPMC_NER
#SBATCH --partition=gpu  
#SBATCH --time=24:00:00        # 8 hours should be plenty for 117k files with 16 workers
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16     # Increasing to 16 to speed up processing
#SBATCH --mail-type=all
#SBATCH --mail-user=malay.gaherwar_singh@tu-dresden.de

# 1. Load Environment
conda init
source ~/miniconda3/etc/profile.d/conda.sh 

conda activate belladonna

# 2. Set Working Directory to the project root
# This ensures "artifacts/..." paths in your script point to the right place
cd /mnt/bulk-saturn/malaygaherwar/belladonna



# 4. Run the script from the scripts folder
python scripts/data_processing_epmc.py