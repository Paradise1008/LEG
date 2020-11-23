#!/bin/bash

#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -p large-gpu
#SBATCH -N 1

#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shine_lsy@gwu.edu

eval "$(conda shell.bash hook)"
conda activate legpy3


cd /CCAS/home/shine_lsy/LEG/methods/mnist
python mnist_kernelSHAP.py

