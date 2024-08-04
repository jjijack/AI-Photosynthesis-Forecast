#!/bin/bash

#SBATCH --job-name=global_forecast
#SBATCH --partition=a100
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=all
#SBATCH --mail-user=sky_jjijack@sjtu.edu.cn
#SBATCH --output=global_forecast.out
#SBATCH --error=global_forecast.err

module load miniconda3
source activate climaX
module load cuda

python src/climax/global_forecast/train.py --config configs/global_forecast_climax.yaml \
    --trainer.strategy=ddp --trainer.devices=1 \
    --trainer.max_epochs=100 \
    --data.root_dir=/dssg/home/acct-kangdj/kangdj-user1/AI-Photosynthesis-Forecast/Data/climax/nc2np_data \
    --data.predict_range=72 \
    --data.batch_size=16 \
    --model.pretrained_path="" \
    --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" \
    --model.weight_decay=1e-5

