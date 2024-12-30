# Brick-by-Brick(BBB)
this is starterkit for [Brick by Brick 2024](https://www.aicrowd.com/challenges/brick-by-brick-2024) challenge.


 
## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. This `preprocess.py` processes time series data for training and testing sets. It handles data loading, preprocessing, and splitting into train/validation sets.
   
#### Data Paths
- `--train_directory` (default: "./train_X/")
  - Directory containing training pickle files
  - Expects pickle files with time series data

- `--test_directory` (default: "./test_X/")
  - Directory containing test pickle files
  - Expects pickle files in the same format as training data

- `--csv_file` (default: "./train_y_v0.1.0.csv")
  - CSV file containing file paths and corresponding labels for training data
  - Should include mapping between files and their target variables

- `--output_directory` (default: "./data/BBB/")
  - Directory where processed data will be saved
  - Will be created if it doesn't exist



Basic usage with default parameters:
```bash
python preprocess.py
```

Custom configuration:
```bash
python preprocess.py \
    --train_directory ./train_X/ \
    --test_directory ./test_X/ \
    --csv_file ./train_y_v0.1.0.csv \
    --output_directory ./processed_data/ \
    --percentile 95 \
    --val_percentage 0.2 \
```



3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following example:

```
python -u run.py   --task_name classification   --is_training 1   --root_path /data1/max/telugu_corpus/time_series/data/BBB   --data_path train.csv \
 --model_id ETTh1_96_96   --model 'TimesNet'   --data BBB   --seq_len 8100   --e_layer 6   --enc_in 1   --d_model 8   --d_ff 32   --des 'Exp6' \
  --itr 1   --top_k 3   --train_epochs 20   --   --num_class 94  --dropout 0.3   --batch_size 32    --learning_rate 0.001
```
4. Colab notebook
   <a href="https://colab.research.google.com/drive/1Bs6aE5gSlM_K0IKH3x2AcvVmuzrpjmzt#scrollTo=DoX7_j6K6T7z" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>



## Acknowledgement

This project is forked from [Tslib](https://github.com/thuml/Time-Series-Library/). We appreciate your contributions and sharing your work with the open-source community.


- Classification: https://www.timeseriesclassification.com/.

