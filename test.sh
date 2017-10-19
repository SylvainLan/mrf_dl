#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=0_python3--u-train.py--batch_size-32--data-pronoun_data_60_20000.pt--save_model-trained--proj_share_weight--external_validation_sc\
ript-validate.sh-pronoun_enfr_dev.lang1-pronoun_enfr_dev.lang2
#SBATCH --gres=gpu:k80:1
#SBATCH --time=160:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --error=/home/sl174/nmt_transformer/LOGS_QSUB/2017-10-10T17:31:12.953064__python3_-u_train.py_-batch_size_32_-data_pronoun_data_60_20000.pt_-\
save_model_trained_-proj_share_weight_-external_validation_script_validate.sh_pronoun_enfr_dev.lang1_pronoun_enfr_dev.lang2/slurm_%j.err
#SBATCH --output=/home/sl174/nmt_transformer/LOGS_QSUB/2017-10-10T17:31:12.953064__python3_-u_train.py_-batch_size_32_-data_pronoun_data_60_20000.pt_\
-save_model_trained_-proj_share_weight_-external_validation_script_validate.sh_pronoun_enfr_dev.lang1_pronoun_enfr_dev.lang2/slurm_%j.out



module purge
module load python3/intel/3.5.3
module load pytorch/python3.5/0.2.0_3


SRC=/home/ss11642/mrf_dl



cd $SRC ; python3 test_cuda.py