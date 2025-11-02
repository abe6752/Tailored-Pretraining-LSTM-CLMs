# Tailored-Pretraining-LSTM-CLMs
---
This repository contains the codes, data preparation scripts, and evaluation workflow used in the following study:

> **Ryuto Abe and Tomoyuki Miyao**,
> * Long Short-term Memory-based Chemical Language Models for Bioactive Molecular Generation Using Tailored Pre-training Datasets*,
> 

## Installation
---
1. Clone this repository:
  
```bash
git clone https://github.com/abe6752/
```

2. Install required packages:
  
```bash
cd Tailored-Pretraining-LSTM-CLMs
conda env create -f environment.yml
conda activate smi-lstm
```

## Dataset Preparation
---
To reproduce the results reported in the paper, you first need to prepare the datasets used for model pre-training and fine-tuning.
  
Please follow the instructions in the Jupyter notebook:
  
👉 `0_dataset_preparation.ipynb`

## Reproducing the Results
---
Once the datasets have been prepared, you can reproduce the experimental results presented in the paper by executing the notebooks in the `notebooks/` directory in the following order:
  
1. `1_smiles_generative_models.ipynb` - Pre-train and fine-tune the LSTM chemical language models
2. `2_pretrain_evaluation.ipynb` - Evaluate pre-training results
3. `3_finetune_evaluation.ipynb` - Evaluate fine-tuning results
  
In the `3_finetune_evaluation.ipynb`, additional tools are required:
- MOSES
- FastTargetPred
- MayaChemTools
  
Please follow the instructions provided in that notebook.