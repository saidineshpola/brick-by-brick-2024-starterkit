# Brick-by-Brick 2024(BBB2024)

this is starterkit for [Brick by Brick 2024](https://www.aicrowd.com/challenges/brick-by-brick-2024) multi label time series classification challenge.

## Usage

### 1. Install Dependencies

Ensure you have Python 3.8 installed. Then, install the required packages by executing the following command:

```sh
pip install -r requirements.txt
```

### 2. Preprocess data

 This `preprocess.py` processes time series data for training and testing sets. It handles data loading, preprocessing, and splitting into train/validation sets.

#### Data Paths

- `--train_directory` (default: "./train_X/")

  - Directory containing training pickle files

- `--test_directory` (default: "./test_X/")

  - Directory containing test pickle files

- `--csv_file` (default: "./train_y_v0.1.0.csv")
  - CSV file containing file paths and corresponding labels for training data

Custom configuration:

```bash
python preprocess.py \
    --train_directory ./train_X/ \
    --test_directory ./test_X/ \
    --csv_file ./train_y_v0.1.0.csv \
    --output_directory ./processed_data/ \
    --val_percentage 0.2 \
```

### 3. Train and evaluate model.
 We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following example:

```
python -u run.py \
    --task_name classification \
    --is_training 1 \
    --d_model 64 \
    --d_ff 128 \
    --root_path /data/time_series/data/BBB \
    --data_path train.csv \
    --model_id ETTh1_96_96 \
    --model 'Transformer' \
    --data BBB \
    --seq_len 336 \
    --e_layer 4 \
    --enc_in 4 \
    --d_model 64 \
    --d_ff 128 \
    --des 'Exp2.2' \
    --itr 1 \
    --top_k 3 \
    --train_epochs 100 \
    --use_multi_gpu \
    --num_class 94 \
    --dropout 0.1 \
    --batch_size 512 \
    --learning_rate 0.001
```
Try experimenting with different settings

### 4. Colab notebook
   <a href="https://colab.research.google.com/drive/1Bs6aE5gSlM_K0IKH3x2AcvVmuzrpjmzt#scrollTo=DoX7_j6K6T7z" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
   
  ## File Structure

  - `data_provider/`: Contains classes for dataset loading, including `data_loader.py`.
  - `exp/`: Contains experiment scripts, including `exp_classification.py` for training the classification model.
  - `preprocess.py`: Script for preprocessing the data before training.
  - `run.py`: Main script to run the training and evaluation.
  - `scripts/`: Contains additional scripts for running experiments.

## Acknowledgement

This project is forked from [Tslib](https://github.com/thuml/Time-Series-Library/). We appreciate your contributions and sharing your work with the open-source community.

- Classification: https://www.timeseriesclassification.com/.
