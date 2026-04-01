# Explainable Smart Building Energy Consumption Forecasting

## Overview

This project focuses on **smart building energy consumption forecasting** using a range of time-series modeling approaches, including traditional statistical methods and deep learning models. The repository organizes experiments around multiple forecasting architectures and provides processed datasets, model artifacts, and baseline outputs for comparison.

The main goal is to explore how different forecasting models perform on building energy consumption data and to support more **interpretable and practical energy prediction workflows** for smart buildings.

## Features

- Building energy consumption forecasting for smart building scenarios
- Comparison of multiple forecasting approaches:
  - ARIMA
  - LSTM
  - Transformer
  - Multi-Head Transformer
- Processed datasets for direct experimentation
- Saved model checkpoints and result plots
- Baseline outputs from additional time-series forecasting models
- Notebook-based workflow for data preparation, training, and evaluation

## Repository Structure

```text
Explainable-Smart-Building-Energy-Consumption-Forecasting/
├── ARIMA/
│   └── am.ipynb
├── lstm/
│   ├── lstm_c.ipynb
│   ├── lstm_m.ipynb
│   ├── best_lstm_model.pth
│   └── lstm_results.png
├── Transformer/
│   ├── 标准tran_Cockatoo.ipynb
│   ├── 标准tran_moose.ipynb
│   ├── best_standard_transformer_model.pth
│   └── standard_transformer_results.png
├── Multi-Head Transformer/
│   ├── tran_v3_Cockatoo_data.ipynb
│   └── tran_v3_Moose_data.ipynb
├── Processed dataset/
│   ├── Cockatoo_data.csv
│   ├── Moose_data.csv
│   └── create.ipynb
└── baseline_outputs/
    ├── Cockatoo_data_Autoformer.pt
    ├── Cockatoo_data_FEDformer.pt
    ├── Cockatoo_data_Informer.pt
    ├── Cockatoo_data_PatchTST.pt
    ├── Moose_data_Autoformer.pt
    ├── Moose_data_FEDformer.pt
    ├── Moose_data_Informer.pt
    ├── Moose_data_PatchTST.pt
    └── baseline_results_summary.csv
```
## Models Included
1. ARIMA

A statistical baseline for time-series forecasting. This model serves as a traditional benchmark against deep learning approaches.

2. LSTM

A recurrent neural network model designed to capture temporal dependencies in sequential energy consumption data.

3. Transformer

A standard Transformer-based forecasting model for learning long-range dependencies in time-series data.

4. Multi-Head Transformer

An extended Transformer architecture using multi-head attention for richer temporal feature representation.

5. Additional Baselines

The repository also includes output artifacts for:

Autoformer

FEDformer

Informer

PatchTST

These outputs can be used for benchmarking and result comparison.

Datasets

The project contains processed datasets in the Processed dataset/ directory:

Cockatoo_data.csv

Moose_data.csv

These datasets appear to be the main data sources used for model training and evaluation.

Workflow

A typical workflow for this project is:

Prepare or inspect the processed datasets

Open the model-specific notebooks

Train or fine-tune forecasting models

Save model checkpoints

Compare outputs and evaluation results across methods

How to Use
1. Clone the repository
git clone https://github.com/cai1157/Explainable-Smart-Building-Energy-Consumption-Forecasting.git
cd Explainable-Smart-Building-Energy-Consumption-Forecasting
2. Set up the environment

It is recommended to create a Python virtual environment first:

python -m venv venv
source venv/bin/activate   # On macOS/Linux
# venv\Scripts\activate    # On Windows

Then install commonly required packages:

pip install numpy pandas matplotlib scikit-learn jupyter torch statsmodels

Note: Depending on the notebook implementation, you may need to install additional packages.

3. Launch Jupyter Notebook
jupyter notebook

Then open the notebook you want to run, for example:

ARIMA/am.ipynb

lstm/lstm_c.ipynb

lstm/lstm_m.ipynb

Transformer/标准tran_Cockatoo.ipynb

Transformer/标准tran_moose.ipynb

Multi-Head Transformer/tran_v3_Cockatoo_data.ipynb

Multi-Head Transformer/tran_v3_Moose_data.ipynb

Explainability

This project is framed as an explainable energy forecasting study. In practical smart building applications, explainability is important because it helps:

understand model behavior over time,

compare forecasting decisions across architectures,

support energy management decisions with more transparency,

improve trust in data-driven building operation systems.

If you want to further strengthen the explainability aspect, you can extend the project with:

attention visualization,

feature importance analysis,

error analysis by time period,

model comparison dashboards.

Results

The repository includes:

saved model weights (.pth)

result plots (.png)

baseline result files (.pt)

a result summary file (baseline_results_summary.csv)

These artifacts support model comparison and reproducibility.

Future Improvements

Add a unified training and evaluation pipeline

Add a requirements.txt file

Add standardized metrics such as MAE, RMSE, and MAPE

Add data schema and feature descriptions

Add visual explain
