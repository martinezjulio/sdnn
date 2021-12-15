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

#CONFIG_FILE='./configs/alexnet/face_inanimate_400k_nobatchnorm.yaml'
#CONFIG_FILE='./configs/alexnet/face_inanimate_400k_orig.yaml'
#CONFIG_FILE='./configs/vgg/face_inanimate_400k_seed2.yaml'
#CONFIG_FILE='./configs/vgg/face_casia_100k_seed1.yaml'
#CONFIG_FILE='./configs/vgg/face_AFD_full_seed.yaml'
#CONFIG_FILE='./configs/vgg/face_na_seed.yaml'
CONFIG_FILE='./configs/vgg/face_AFD_matched_seed.yaml'
#CONFIG_FILE='./configs/vgg/cars_seed.yaml'
#CONFIG_FILE='./configs/resnet/face_400k_resnet50.yaml'
#CONFIG_FILE='./configs/resnet/inanimate_400k_resnet50.yaml'
#CONFIG_FILE='./configs/alexnet/face_400k_nobatchnorm.yaml'
#CONFIG_FILE='./configs/alexnet/inanimate_400k_nobatchnorm.yaml'

SCRIPT=./train_new.py

hostname
date
echo "Sourcing conda..."
source /mindhive/nklab4/users/kdobs/anaconda3/bin/activate
date
echo "Activating conda env..."
#conda activate test
conda activate torch-gpu-dev
date
echo "Running python script..."
CUDA_VISIBLE_DEVICES=0 python $SCRIPT --config_file $CONFIG_FILE --num_epochs 201 --read_seed 1 --maxout True --save_freq 10 --valid_freq 1 --use_scheduler "True" --pretrained "False" # --custom_learning_rate 0.0001

date
echo "Job completed"
