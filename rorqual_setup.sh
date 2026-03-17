#!/bin/bash
#SBATCH --gpus=h100_3g.40gb:1      
#SBATCH --cpus-per-task=8  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=62000M       
#SBATCH --time=0-05:00     # DD-HH:MM:SS
#SBATCH --job-name=webcam-resnet
#SBATCH --output=webcam-resnet-%J.out

PROJECT=~/links/projects/def-boakes/garnierk/
SOURCEDIR=$PROJECT/webcam-avalanche-detection/
export TORCH_HOME=$SOURCEDIR/torch_cache

cd $SOURCEDIR

module purge

module load gcc opencv/4.8.1 python/3.10 scipy-stack/2024a cuda cudnn httpproxy

cd $SLURM_TMPDIR
virtualenv --no-download .venv

source .venv/bin/activate

# Install dependencies
#
cd $SOURCEDIR

#pip install --upgrade pip
python -m pip install --no-index -r requirements.txt

python -m pip install -e .

mkdir $SLURM_TMPDIR/data
unzip -q -d $SLURM_TMPDIR/data uibk_avalanches.zip

python utils/train_test_split.py --source-dir $SLURM_TMPDIR/data --output-dir $SLURM_TMPDIR/data

python -m classification.experiments.benchmarking \
  --data-dir $SLURM_TMPDIR/data \
  --project-dir $SOURCEDIR
 

# Install PyTorch
# The exact version required is system-dependent:
# see https://pytorch.org/
#python -m pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
