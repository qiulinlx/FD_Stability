#!/bin/bash

# ============================
# SGE Job Configuration Block
# ============================

#$ -l h_rt=10:0:0
#$ -l tmem=10G
#$ -wd /home/qiulinli/FD_Stability
#$ -N NeuralNet
#$ -l gpu=1
#$ -P pnlab
#$ -j y
#$ -S /bin/bash

# ============================
# Execution Section
# ============================

cd /home/qiulinli/FD_Stability

echo "Running on:"
hostname

echo "Start time:"
date

echo "GPU info:"
nvidia-smi

# Activate environment
source /home/qiulinli/FD_Stability/dl_env/bin/activate

echo "Python path:"
which python

# Run script
python nn_main.py

echo "End time:"
date