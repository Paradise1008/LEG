#!/bin/bash

#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -p debug-gpu
#SBATCH -N 1

#SBATCH -t 4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shine_lsy@gwu.edu

eval "$(conda shell.bash hook)"
conda activate legpy3


cd /CCAS/home/shine_lsy/LEG/methods
python LEGv0.py

