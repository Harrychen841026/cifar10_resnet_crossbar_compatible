#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH -G 1
#SBATCH -t 8:00:00

ml load py-tensorflow
ml load py-keras/2.3.1_py36

for reg in 0 0.00001 0.00005 0.0001
do
  srun python3 cifar10_resnet_BN.py 3 0 0 $reg
done

