#!/bin/bash


## CPU
# SBATCH --qos=nf
# SBATCH --mem-per-cpu=62Gb
# SBATCH --time=24:00:00

#SBATCH --job-name=Gradient_Importance
#SBATCH --output=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Gradient_Importance.out
#SBATCH --error=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Gradient_Importance_Error.out

module load python3

python3 /home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Gradient_Importance.py