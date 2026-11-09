#!/bin/bash

#SBATCH --qos=nf
#SBATCH --mem-per-cpu=62Gb
#SBATCH --time=48:00:00
#SBATCH --job-name=Dataloading
#SBATCH --output=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Making_Tensors.out
#SBATCH --error=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Making_Tensors_Error.out

module load python3

python3 /home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Making_Tensors.py
