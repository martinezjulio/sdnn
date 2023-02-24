#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mail-type=END
#SBATCH --mail-user=kdobs@mit.edu

#SBATCH --partition=nklab
##SBATCH --partition=normal

#SBATCH --job-name=vgg_afd

#SBATCH --time=6-23:00:00

##SBATCH --gres=gpu:GEFORCERTX2080TI:1

##SBATCH --gres=gpu:titan-x:1
#SBATCH --gres=gpu:QUADRORTX6000:1

#SBATCH --mem=12G

#SBATCH --output='./output/train/%A_%a.out'

CONFIG_FILE='./configs/vgg/face_dual_whitasia.yaml'


SCRIPT=./train.py

hostname
date
echo "Sourcing python..."
source /shared/venvs/py3.8-torch1.7.1/bin/activate
date
echo "Activating required packages..."
#conda activate test
#conda activate torch-gpu-dev
pip install -r requirements.txt
date
echo "Running python script..."
CUDA_VISIBLE_DEVICES=0 python $SCRIPT --config_file $CONFIG_FILE --num_epochs 201 --read_seed 1 --maxout True --save_freq 10 --valid_freq 1 --use_scheduler "True" --pretrained "False" # --custom_learning_rate 0.0001

date
echo "Job completed"
