#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=0_python3--test_cuda.py
ript-validate.sh-pronoun_enfr_dev.lang1-pronoun_enfr_dev.lang2
#SBATCH --gres=gpu:k80:1
#SBATCH --time=00:10:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --error=/home/ss11642/mrf_dl
save_model_trained_-proj_share_weight_-external_validation_script_validate.sh_pronoun_enfr_dev.lang1_pronoun_enfr_dev.lang2/slurm_%j.err
#SBATCH --output=/home/ss11642/mrf_dl
-save_model_trained_-proj_share_weight_-external_validation_script_validate.sh_pronoun_enfr_dev.lang1_pronoun_enfr_dev.lang2/slurm_%j.out



module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3


SRC=/home/ss11642/mrf_dl



cd $SRC ; python3 test_cuda.py