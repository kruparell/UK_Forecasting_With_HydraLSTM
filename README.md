# Hydra-LSTM for Reservoir-Influenced Streamflow Prediction

A machine learning framework for simulating daily river discharge in the UK, with a focus on reservoir-dominated catchments using the Hydra-LSTM architecture.

## Overview
This repository implements and compares multiple LSTM-based approaches for streamflow prediction in catchments with varying degrees of reservoir influence. The Hydra-LSTM architecture uses a multi-head design where a shared body is trained on natural catchments and specialized heads are trained for reservoir-influenced systems.

## Dataset
This project uses the **CAMELS-GB** dataset (Coxon et al.), which provides:
- Meteorological forcings
- Catchment attributes
- Streamflow observations for UK catchments
- Reservoir influence indicators

## Model Architectures

### 1. **Full LSTM**
- Trained on both reservoir-influenced and natural catchments
- Single unified model for all catchment types

### 2. **Natural LSTM**
- Trained exclusively on non-reservoir catchments
- Baseline model for natural streamflow dynamics

### 3. **Reservoir LSTM**
- Trained exclusively on reservoir-influenced catchments
- Specialized model for reservoir-dominated systems

### 4. **Hydra-LSTM**
- **Body**: Shared LSTM layers trained on natural catchments
- **Heads**: Specialized output layers for different catchment types
  - Main head: For general predictions
  - Reservoir embedding head: For reservoir-influenced catchments
- Leverages transfer learning from natural to regulated systems

## Repository Structure

```
.
├── CamelsGB_Models/          # Trained model checkpoints
│   ├── Full/                 # Full LSTM models
│   ├── Hydra/                # Hydra-LSTM body and heads
│   ├── Natural/              # Natural catchment models
│   └── Reservoir/            # Reservoir catchment models
│
├── Catchment_Lists/          # Train/validation/test splits
│   ├── Reservoirs_*.txt      # Reservoir catchment IDs
│   ├── NonReservoirs_*.txt   # Natural catchment IDs
│   └── basins_and_dates.pkl  # Temporal metadata
│
├── ML_Functions/             # Core ML utilities
│   ├── ML_functions.py       # Data loading and preprocessing
│   ├── ML_Models.py          # Model architectures
│   └── ML_Training.py        # Training loops
│
├── UK_Batch_Scripts/         # HPC training scripts
│   ├── *_Hyperparameter_Training.py   # Hyperparameter search
│   ├── *_Hyperparameter_Training.sh   # SLURM job scripts
│   └── Output_Files/         # Training logs
│
├── Sample_Pairs/             # Preprocessed training pairs
├── Plots/                    # Model evaluation plots
│
├── 1Exploring_Reservoirs.ipynb           # Data exploration
├── 2Normalising_Temporal_Data.ipynb      # Preprocessing
└── 3Training_LSTMs.ipynb                 # Training notebook│
```

## Model Naming Convention

Model files follow the pattern:
```
{ModelType}_LSTM_Numlayers{N}_Embhs{H}_Embos{O}[_BodyOut{B}][_Component].pt
```

- **ModelType**: Full, Natural, Reservoir, or Hydra
- **Numlayers**: Number of LSTM layers (1 or 2)
- **Embhs**: Embedding hidden size (e.g., 64, 128)
- **Embos**: Embedding output size (e.g., 8, 16, 32)
- **BodyOut**: Body output size (Hydra only)
- **Component**: Body, Main_Head, or Res_Embedding_Head (Hydra only)

**Example**: `Hydra_LSTM_Numlayers1_Embhs64_Embos16_BodyOut32_Body.pt`
- 1-layer Hydra LSTM body
- Embedding hidden size: 64
- Embedding output size: 16
- Body output size: 32

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hydra-lstm-reservoir.git
cd hydra-lstm-reservoir

# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- Python 3.11+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Usage

### 1. Data Preprocessing
```bash
jupyter notebook 2Normalising_Temporal_Data.ipynb
```

### 2. Training Models

**Local training** (Jupyter):
```bash
jupyter notebook 3Training_LSTMs.ipynb
```

**HPC training** (SLURM):
```bash
cd UK_Batch_Scripts
sbatch Natural_Hyperparameter_Training.sh
sbatch Reservoir_Hyperparameter_Training.sh
sbatch Hydra_Hyperparameter_Training.sh
sbatch Full_Hyperparameter_Training.sh
```

## Evaluation Metrics

Models are evaluated using:
- **NSE** (Nash-Sutcliffe Efficiency)
- **KGE** (Kling-Gupta Efficiency)
- **KGE components**: Correlation (r), Bias (β), Variability (α)

Results are visualized through:
- Cumulative distribution functions (CDFs)
- Hydrograph comparisons

## Catchment Splits

Catchments are divided based on reservoir influence:
- **Splits**: Train, Validation, Test for both natural and reservoir catchments

## Key Features

- Multiple LSTM architectures for comparison
- Hydra multi-head design for transfer learning
- Comprehensive hyperparameter search
- SLURM-compatible HPC scripts
- Normalized temporal data preprocessing
- Stratified train/val/test splits
- Extensive evaluation metrics and visualizations

## Project Status

**In Development** - Initial commit for ongoing research

This is an active research project. A manuscript is in preparation.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hydra_lstm_reservoir_2025,
  author = {Karan Ruparell},
  title = {Hydra-LSTM for Reservoir-Influenced Streamflow Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/hydra-lstm-reservoir}
}
```

**CAMELS-GB Dataset**:
```bibtex
@article{coxon2020camels,
  title={CAMELS-GB: hydrometeorological time series and landscape attributes for 671 catchments in Great Britain},
  author={Coxon, Gemma and Addor, Nans and Bloomfield, John P and Freer, Jim and Fry, Matt and Hannaford, Jamie and Howden, Nicholas JK and Lane, Rosanna and Lewis, Melinda and Robinson, Emma L and others},
  journal={Earth System Science Data},
  volume={12},
  number={4},
  pages={2459--2483},
  year={2020}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This code has an MIT open license

## Contact

For questions or collaboration:
- **Email**: Karan.ruparell2@pgr.reading.ac.uk
- **Issues**: [GitHub Issues](https://github.com/yourusername/hydra-lstm-reservoir/issues)

## Acknowledgments

- CAMELS-GB dataset providers (Coxon et al.)
- AFESP DTP Programme
- Supervisors: Kieran M. R. Hunt, Hannah L. Cloke,
  Christel Prudhomme, Florian Pappenberger, Matthew Chantry

---

**Note**: This repository is part of ongoing research. Code and models are subject to updates as the project develops.
