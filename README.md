# Time Series Library (TSLib)
TSLib is an open-source library for deep learning researchers, especially for deep time series analysis.


 
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

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following example:

```
python -u run.py   --task_name classification   --is_training 1   --root_path /data1/max/telugu_corpus/time_series/data/BBB   --data_path train.csv \
 --model_id ETTh1_96_96   --model 'TimesNet'   --data BBB   --seq_len 8100   --e_layer 6   --enc_in 1   --d_model 8   --d_ff 32   --des 'Exp6' \
  --itr 1   --top_k 3   --train_epochs 20   --   --num_class 94  --dropout 0.3   --batch_size 32    --learning_rate 0.001
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

Note: 

(1) About classification: Since we include all five tasks in a unified code base, the accuracy of each subtask may fluctuate but the average performance can be reproduced (even a bit better). We have provided the reproduced checkpoints [here](https://github.com/thuml/Time-Series-Library/issues/494).


## Citation

If you find this repo useful, please cite this paper.

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}

@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```

## Contact
If you have any questions or suggestions, feel free to contact our maintenance team:

Current:
- Haixu Wu (Ph.D. student, wuhx23@mails.tsinghua.edu.cn)
- Yong Liu (Ph.D. student, liuyong21@mails.tsinghua.edu.cn)
- Yuxuan Wang (Ph.D. student, wangyuxu22@mails.tsinghua.edu.cn)
- Huikun Weng (Undergraduate, wenghk22@mails.tsinghua.edu.cn)

Previous:
- Tengge Hu (Master student, htg21@mails.tsinghua.edu.cn)
- Haoran Zhang (Master student, z-hr20@mails.tsinghua.edu.cn)
- Jiawei Guo (Undergraduate, guo-jw21@mails.tsinghua.edu.cn)

Or describe it in Issues.

## Acknowledgement

This project is supported by the National Key R&D Program of China (2021YFB1715200).

This library is constructed based on the following repos:

- Forecasting: https://github.com/thuml/Autoformer.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://github.com/thuml/Flowformer.

All the experiment datasets are public, and we obtain them from the following links:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer.

- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://www.timeseriesclassification.com/.

## All Thanks To Our Contributors

<a href="https://github.com/thuml/Time-Series-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=thuml/Time-Series-Library" />
</a>
