import os
from pickle import dump
import pickle
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
import joblib
import random
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")


def time_series_augmentations(x, y, cut_ratio=0.4, num_augmentations=1):
    """
    Perform multiple time series augmentations to generate additional training data.

    Args:
        x (np.ndarray): Input time series data of shape (sequence_length, features)
        y (np.ndarray): Labels
        cut_ratio (float): Ratio of sequence to cut (default: 0.4)
        num_augmentations (int): Number of augmented samples to generate

    Returns:
        tuple: Lists of augmented sequences and corresponding labels
    """
    augmented_x = []
    augmented_y = []

    for _ in range(num_augmentations):
        # Randomly choose augmentation method
        aug_method = random.choice(
            ["cut_and_resize", "add_noise", "time_warp", "magnitude_warp"]
        )

        if aug_method == "cut_and_resize":
            aug_x = cut_and_resize(x, cut_ratio)
        elif aug_method == "add_noise":
            aug_x = add_noise(x, noise_level=0.05)
        elif aug_method == "time_warp":
            aug_x = time_warp(x, sigma=0.2)
        else:
            aug_x = magnitude_warp(x, sigma=0.2)

        augmented_x.append(aug_x)
        augmented_y.append(y)

    return augmented_x, augmented_y


def cut_and_resize(x, cut_ratio=0.4):
    """Cut a portion of the sequence and resize back to original length."""
    seq_len = len(x)
    cut_length = int(seq_len * cut_ratio)
    start_idx = random.randint(0, seq_len - cut_length)

    # Cut the sequence
    mask = np.ones(seq_len, dtype=bool)
    mask[start_idx : start_idx + cut_length] = False
    reduced_seq = x[mask]

    # Resize back to original length using interpolation
    original_indices = np.arange(len(reduced_seq))
    new_indices = np.linspace(0, len(reduced_seq) - 1, seq_len)

    if len(x.shape) > 1:  # Handle multivariate time series
        aug_x = np.zeros_like(x)
        for i in range(x.shape[1]):
            f = interp1d(original_indices, reduced_seq[:, i], kind="linear")
            aug_x[:, i] = f(new_indices)
    else:
        f = interp1d(original_indices, reduced_seq, kind="linear")
        aug_x = f(new_indices)

    return aug_x


def add_noise(x, noise_level=0.05):
    """Add random noise to the sequence."""
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise


def time_warp(x, sigma=0.2):
    """Apply time warping by adjusting the time steps."""
    seq_len = len(x)
    warp = np.random.normal(loc=1.0, scale=sigma, size=(seq_len,))
    cumulative_warp = np.cumsum(warp)
    warped_time = cumulative_warp / cumulative_warp[-1] * (seq_len - 1)

    if len(x.shape) > 1:  # Handle multivariate time series
        aug_x = np.zeros_like(x)
        for i in range(x.shape[1]):
            f = interp1d(np.arange(seq_len), x[:, i], kind="linear")
            aug_x[:, i] = f(warped_time)
    else:
        f = interp1d(np.arange(seq_len), x, kind="linear")
        aug_x = f(warped_time)

    return aug_x


def magnitude_warp(x, sigma=0.2):
    """Apply magnitude warping by scaling the values."""
    if len(x.shape) > 1:  # Handle multivariate time series
        aug_x = np.zeros_like(x)
        for i in range(x.shape[1]):
            scale = np.random.normal(loc=1.0, scale=sigma)
            aug_x[:, i] = x[:, i] * scale
    else:
        scale = np.random.normal(loc=1.0, scale=sigma)
        aug_x = x * scale

    return aug_x


def run_augmentation_single(x, y, args):
    """
    Run augmentation for a single sequence.

    Args:
        x (np.ndarray): Input sequence
        y (np.ndarray): Label
        args: Arguments containing augmentation parameters

    Returns:
        tuple: Augmented sequence, label, and augmentation info
    """
    aug_x, aug_y = time_series_augmentations(x, y, cut_ratio=0.4, num_augmentations=1)

    return aug_x[0], aug_y[0], {"method": "random_augmentation"}



class Dataset_BBB(Dataset):
    """
    Custom Dataset class for loading and preprocessing data.

    Attributes:
        args: Arguments passed to the dataset.
        root_path: Root directory path where data files are stored.
        flag: Indicates whether the dataset is for training, validation, or testing.
        features: Type of features to be used.
        scale: Boolean indicating whether to apply scaling.
        timeenc: Time encoding flag.
        freq: Frequency of the data.
        seed: Random seed for reproducibility.
        val_ratio: Ratio of validation data.
        data_x: List of input data series.
        data_y: List of labels corresponding to the input data.
        file_names: List of file names (only for test data).
    """

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        scaling=True,
        timeenc=0,
        freq="h",
        seed=42,
        val_ratio=0.2,
    ):
        """
        Initializes the Dataset_BBB object with the given parameters.

        Args:
            args: Arguments passed to the dataset.
            root_path: Root directory path where data files are stored.
            flag: Indicates whether the dataset is for training, validation, or testing.
            size: Size of the dataset (not used in this implementation).
            features: Type of features to be used.
            scaling: Boolean indicating whether to apply scaling.
            timeenc: Time encoding flag.
            freq: Frequency of the data.
            seed: Random seed for reproducibility.
            val_ratio: Ratio of validation data.
        """
        self.args = args
        self.features = features
        self.scale = scaling
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag.lower()
        self.root_path = root_path
        self.seed = seed
        self.val_ratio = val_ratio
        np.random.seed(seed)
        self.__read_data__()

    def print_data_stats(self, data, stage):
        """
        Helper function to print data statistics.

        Args:
            data: Data for which statistics are to be printed.
            stage: Stage of the data (e.g., 'train', 'val', 'test').

        Returns:
            Dictionary containing statistics of the data.
        """
        if isinstance(data, list):
            data = np.concatenate(data)
        stats = {
            "min": np.min(data),
            "max": np.max(data),
            "mean": np.mean(data),
            "std": np.std(data),
            "nan_count": np.isnan(data).sum(),
            "inf_count": np.isinf(data).sum(),
        }
        print(f"\n{stage} statistics: {stats}")
        return stats

    def preprocess_data(self, data):
        """
        Apply selective log transformation to data.

        Args:
            data: Input data to be preprocessed.

        Returns:
            Preprocessed data.
        """
        data = np.array(data, dtype=np.float32)
        transformed = np.zeros_like(data)

        # Store signs and work with absolute values
        signs = np.sign(data)
        abs_data = np.abs(data)

        # For values with magnitude > 10
        mask_high = abs_data > 10
        transformed[mask_high] = np.log(abs_data[mask_high] - 10 + 1) + np.log(10)

        # Keep values between -10 and 10 unchanged
        mask_mid = abs_data <= 10
        transformed[mask_mid] = data[mask_mid]

        # Handle any NaN or inf values
        transformed = np.nan_to_num(
            transformed, nan=0.0, posinf=np.log(1e10), neginf=-np.log(1e10)
        )

        return transformed * signs

    def inverse_transform(self, data):
        """
        Inverse transform the preprocessing.

        Args:
            data: Preprocessed data to be inverse transformed.

        Returns:
            Original data before preprocessing.
        """
        data = np.array(data)
        signs = np.sign(data)
        abs_data = np.abs(data)

        transformed = np.zeros_like(data, dtype=np.float32)

        # Inverse transform for values that were > 10
        mask_high = abs_data > np.log(10)
        transformed[mask_high] = (
            np.exp(abs_data[mask_high] - np.log(10)) + 10 - 1
        ) * signs[mask_high]

        # Values <= log(10) were unchanged
        mask_low = abs_data <= np.log(10)
        transformed[mask_low] = data[mask_low]

        return transformed

    def __read_data__(self):
        """
        Reads data from files and applies preprocessing.
        """
        try:
            if self.flag == "test":
                data_path = os.path.join(self.root_path, "test_data.pkl")
                with open(data_path, "rb") as f:
                    data_dict = pickle.load(f)

                self.data_x = data_dict["series"]
                self.file_names = data_dict["file_names"]

                # Apply preprocessing
                for i in range(len(self.data_x)):
                    self.data_x[i] = self.preprocess_data(self.data_x[i])

            else:  # train or val
                split_name = "train" if self.flag == "train" else "val"
                data_path = os.path.join(self.root_path, f"{split_name}_data.pkl")

                with open(data_path, "rb") as f:
                    data_dict = pickle.load(f)

                self.data_x = data_dict["series"]
                self.data_y = data_dict["labels"]
                self.data_y[self.data_y == -1.0] = 0.0  # Convert -1 to 0

                # Apply preprocessing
                for i in range(len(self.data_x)):
                    self.data_x[i] = self.preprocess_data(self.data_x[i])

                if self.flag == "train":
                    # Generate shuffle indices
                    shuffle_idx = np.random.permutation(len(self.data_x))
                    self.data_x = [self.data_x[i] for i in shuffle_idx]
                    self.data_y = [self.data_y[i] for i in shuffle_idx]

        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            raise

    def __getitem__(self, index):
        """
        Retrieves a single data point from the dataset.

        Args:
            index: Index of the data point to retrieve.

        Returns:
            Tuple of input data and corresponding label (or dummy label for test data).
        """
        try:
            seq_x = self.data_x[index]
            if len(seq_x.shape) == 1:
                seq_x = np.expand_dims(seq_x, axis=1)

            if self.flag == "test":
                return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(
                    torch.zeros([1, 94]), dtype=torch.float32
                )
            else:
                seq_y = self.data_y[index]
                if (
                    hasattr(self.args, "augmentation_ratio")
                    and self.args.augmentation_ratio > 0
                    and self.flag == "train"
                ):
                    seq_x, seq_y, _ = run_augmentation_single(seq_x, seq_y, self.args)

                return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(
                    seq_y, dtype=torch.float32
                )
        except Exception as e:
            print(f"Error in __getitem__ for index {index}: {str(e)}")
            raise

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.data_x)

    def get_label_distribution(self):
        """
        Helper method to get the distribution of labels.

        Returns:
            Sum of labels across the dataset.
        """
        if hasattr(self, "data_y"):
            return np.sum(self.data_y, axis=0)
        return None

# Sample collate_fn
def collate_fn(batch, max_len):
    batch_x, batch_y = zip(*batch)
    max_seq_len = max(len(x) for x in batch_x)
    padded_x = torch.zeros((len(batch_x), max_seq_len, batch_x[0].shape[-1]))
    padded_y = torch.zeros((len(batch_y), max_len, batch_y[0].shape[-1]))

    for i in range(len(batch_x)):
        end = len(batch_x[i])
        padded_x[i, :end, :] = batch_x[i]
        padded_y[i, : len(batch_y[i]), :] = batch_y[i]

    return padded_x, padded_y


class Dataset_Custom(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(
                self.data_x, self.data_y, self.args
            )

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
