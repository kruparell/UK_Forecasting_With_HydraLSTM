#!/bin/bash

#SBATCH --qos=nf
#SBATCH --mem-per-cpu=62Gb
#SBATCH --time=48:00:00
#SBATCH --job-name=MetaDataloading
#SBATCH --output=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Making_MetaTensors.out
#SBATCH --error=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Making_MetaTensors_Error.out

module load python3

python3 /home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Getting_tensor_metadata.py
