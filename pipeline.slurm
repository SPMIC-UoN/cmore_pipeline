#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --job-name=t1_seg
#SBATCH --partition=imgcomputeq,imghmemq
#NOTSBATCH --partition=imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32g
#SBATCH --qos=img
#SBATCH --array=1-441
#NOTSBATCH --gres=gpu:1
#SBATCH --export=NONE

module load conda-img
module load dcm2niix-img
module load renal-preproc-img
module load quantiphyse-img
source activate img

python -u cmore_pipeline.py --input /spmstore/project/RenalMRI/cmore/data/ --output cmore_output2 --subjid-idx=${SLURM_ARRAY_TASK_ID} --skip-preproc --skip-seg


