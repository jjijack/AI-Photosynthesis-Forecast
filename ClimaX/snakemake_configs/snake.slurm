#!/bin/bash

#SBATCH --job-name=snake
#SBATCH --partition=64c512g
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3
source activate snakemake
cd HAMMOZ/ 
snakemake all --configfile config_specific_humidity.yml --cores 2
