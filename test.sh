#!/usr/bin/bash

<<<<<<< HEAD
#SBATCH -J multi_exit_resnet
=======
#SBATCH -J hierFL
>>>>>>> 2e63113b9161ebf69c7a6465ea62bc98918f052c
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=28G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

<<<<<<< HEAD
which python
hostname
python multi_exit_resnet.py
exit 0
=======

hostname
python HierFL/hierfavg.py
exit 0

>>>>>>> 2e63113b9161ebf69c7a6465ea62bc98918f052c
