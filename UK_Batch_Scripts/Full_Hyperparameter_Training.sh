#!/bin/bash

#SBATCH --qos=nf
#SBATCH --mem-per-cpu=62Gb
#SBATCH --time=24:00:00
#SBATCH --job-name=Full_Hyperparameter_Training
#SBATCH --output=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Full_Hyperparameter_Training.out
#SBATCH --error=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Full_Hyperparameter_Training_Error.out

module load python3

python3 /home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Full_Hyperparameter_Training.py