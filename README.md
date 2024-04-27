# Home-Credit---Credit-Risk-Model-Statibility
ISYE-6740 GT

## Runbook

### Prepare Data and Train Model
1. run `data/prepare_data.py` script to generate a single parquet file from which to split into train and test subset. The script will create new directories locally and store the output in `data/train/` and `data/test/`
2. all models are stored in `model/`. Running `model/lgmb.py` script will train LightGradientBoost model. This script will load data from `data/train/`
