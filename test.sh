#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=0_python3--test_cuda.py
ript-validate.sh-pronoun_enfr_dev.lang1-pronoun_enfr_dev.lang2
#SBATCH --gres=gpu:k80:1
#SBATCH --time=01:00:00
#SBATCH --mem=2GB
#SBATCH --nodes=1
#SBATCH --output=/home/ss11642/mrf_dl/slurm_%j.out

module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3


SRC=/home/ss11642/mrf_dl



cd $SRC

python3 test_cuda.py