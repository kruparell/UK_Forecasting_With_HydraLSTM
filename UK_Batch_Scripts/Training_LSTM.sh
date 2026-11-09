#!/bin/bash

#SBATCH --qos=nf
#SBATCH --mem-per-cpu=62Gb
#SBATCH --time=24:00:00
#SBATCH --job-name=Dataloading
#SBATCH --output=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Training_LSTM.out
#SBATCH --error=/home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Output_Files/Training_LSTM_Error.out

module load python3

python3 /home/mokr/Reservoir_Modelling/UK_Batch_Scripts/Training_LSTM.py