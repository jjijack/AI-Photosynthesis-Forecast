#!/bin/bash

#SBATCH --job-name=preprocessing
#SBATCH --partition=64c512g
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH --output=preprocessing.out
#SBATCH --error=preprocessing.err

module load miniconda3
source activate climaX
python src/data_preprocessing/nc2np_equally_cmip6.py \
    --dataset hammoz \
    --path ~/AI-Photosynthesis-Forecast/Data/climax/CMIP6/HAMMOZ/5.625deg/ \
    --num_shards 10 \
    --save_dir ~/AI-Photosynthesis-Forecast/Data/climax/CMIP6/HAMMOZ/5.625deg_np_10shards
