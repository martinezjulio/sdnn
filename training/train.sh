#!/bin/bash

#SBATCH --nodes=1    # Each node has 16 or 20 CPU cores.
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Elaheh.Akbarifathkouhi@psychol.uni-giessen.de

#SBATCH --partition=single
##SBATCH --partition=normal

#SBATCH --job-name=vgg_DualRace

#SBATCH --time=4-23:00:00

#SBATCH --gres=gpu:1

#SBATCH --mem=12G

#SBATCH --output='/home/elaheh_akbari/sdnn-otherrace/output/train/%A_%a.out'

CONFIG_FILE='./configs/vgg/face_dual_whitasia.yaml'


SCRIPT=/home/elaheh_akbari/sdnn-otherrace/training/train.py

hostname
date
echo "Sourcing python..."
source /shared/venvs/py3.8-torch1.7.1/bin/activate
date
echo "Installing requirements..."

#virtualenv project_otherrace
#source project_otherrace/bin/activate
pip install -r requirements.txt
date
echo "Running python script..."
CUDA_VISIBLE_DEVICES=0,1 python $SCRIPT --config_file $CONFIG_FILE --num_epochs 201 --read_seed 1 --maxout True --save_freq 10 --valid_freq 1 --use_scheduler "True" --pretrained "False" # --custom_learning_rate 0.0001

date
echo "Job completed"
