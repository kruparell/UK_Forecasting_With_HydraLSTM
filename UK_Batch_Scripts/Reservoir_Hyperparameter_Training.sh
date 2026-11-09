#!/bin/bash


## CPU
## SBATCH --qos=nf
## SBATCH --mem-per-cpu=62Gb
## SBATCH --time=48:00:00


## GPU
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00


#SBATCH --job-name=Reservoir_Hyperparameter_Training_NoFlow
#SBATCH --output=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Reservoir_Hyperparameter_Training_NoLat_NoFlow.out
#SBATCH --error=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Reservoir_Hyperparameter_Training_NoLat_NoFlow_Error.out

module load python3

python3 /home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Reservoir-Hyperparameter-Training-Refactored.py