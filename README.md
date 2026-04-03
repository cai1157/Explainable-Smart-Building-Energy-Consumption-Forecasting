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

# dependencies:
  - python=3.10
  - numpy=1.26.4
  - pandas=2.2.2
  - matplotlib=3.8.4
  - scikit-learn=1.4.2
  - scipy=1.13.1
  - statsmodels=0.14.2
  - jupyter=1.0.0
  - notebook=7.2.1
  - ipykernel=6.29.5
  - pip
  - pytorch=2.3.1
  - torchvision=0.18.1
  - torchaudio=2.3.1
  - pip:
      - typing-extensions>=4.8
# Input Feature Description

## Overview

This project uses **43 input features** for single-step building energy consumption forecasting.
The target variable is the building electricity consumption column in each dataset:

- `Cockatoo_education_Erik` for `Cockatoo_data.csv`
- `Moose_education_Ricardo` for `Moose_data.csv`

The raw input files contain:
- one target energy column,
- `airTemperature`,
- `windSpeed`,
- and a `timestamp` column used to derive calendar features.

The final 43 input features are grouped into three categories:
- Environmental features: 2
- Time-related features: 36
- Historical energy features: 5

## Feature Construction Rules

### 1. Environmental features
Directly taken from the processed dataset:

1. `airTemperature`  
   Outdoor air temperature.

2. `windSpeed`  
   Outdoor wind speed.

### 2. Time-related features

#### 2.1 Weekday one-hot features (7)
Derived from `timestamp.dayofweek`:

3. `weekday_0` — Monday  
4. `weekday_1` — Tuesday  
5. `weekday_2` — Wednesday  
6. `weekday_3` — Thursday  
7. `weekday_4` — Friday  
8. `weekday_5` — Saturday  
9. `weekday_6` — Sunday  

#### 2.2 Weekend indicator (1)
10. `is_weekend_1` — 1 if Saturday/Sunday, else 0

#### 2.3 Hour one-hot features (24)
Derived from `timestamp.hour`:

11. `hour_0`  
12. `hour_1`  
13. `hour_2`  
14. `hour_3`  
15. `hour_4`  
16. `hour_5`  
17. `hour_6`  
18. `hour_7`  
19. `hour_8`  
20. `hour_9`  
21. `hour_10`  
22. `hour_11`  
23. `hour_12`  
24. `hour_13`  
25. `hour_14`  
26. `hour_15`  
27. `hour_16`  
28. `hour_17`  
29. `hour_18`  
30. `hour_19`  
31. `hour_20`  
32. `hour_21`  
33. `hour_22`  
34. `hour_23`

#### 2.4 Season one-hot features (4)
Derived from `timestamp.month` using:
- Spring = 0 → March, April, May
- Summer = 1 → June, July, August
- Autumn = 2 → September, October, November
- Winter = 3 → December, January, February

35. `season_0` — Spring  
36. `season_1` — Summer  
37. `season_2` — Autumn  
38. `season_3` — Winter  

### 3. Historical energy features (5)
Constructed from the target energy series:

39. `lag_1h`  
    Energy consumption at the previous hour.

40. `lag_24h`  
    Energy consumption at the same hour on the previous day.

41. `lag_168h`  
    Energy consumption at the same hour one week earlier.

42. `roll_mean_24h`  
    Rolling mean of the previous 24 hours.

43. `roll_max_24h`  
    Rolling maximum of the previous 24 hours.

## Target Variable

The target is **not included as the current-time input feature**.
Instead, historical information from the target series is introduced through lag and rolling statistics.

## Preprocessing Notes

- Missing values created by lag/rolling operations are filled using forward-fill and backward-fill.
- Features are normalized with `MinMaxScaler(feature_range=(0, 1))`.
- The target variable is also normalized separately with `MinMaxScaler(feature_range=(0, 1))`.
- A sliding window of **24 time steps** is used:
  - input: previous 24 hours,
  - output: next 1 hour energy consumption.
- Dataset split is chronological:
  - train: 80%
  - validation: 10%
  - test: 10%
