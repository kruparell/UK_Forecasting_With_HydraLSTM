#!/bin/bash


## CPU
# SBATCH --qos=nf
# SBATCH --mem-per-cpu=62Gb
# SBATCH --time=24:00:00

#SBATCH --job-name=Evaluation_Script
#SBATCH --output=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Evaluation_Script.out
#SBATCH --error=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Evaluation_Script_Error.out

module load python3

python3 /home/mokr/Reservoir_Modelling/UK_Batch_Scripts/run_evaluation.py