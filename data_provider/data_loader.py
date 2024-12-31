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


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dataset_BBB(Dataset):
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
        self.args = args
        self.features = features
        self.scale = scaling
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag.lower()
        self.root_path = root_path
        self.seed = seed
        self.val_ratio = val_ratio
        self.kept_indices = [5, 12, 54, 65, 66, 71]
        np.random.seed(seed)
        self.scaler = None
        self.scaler_path = os.path.join(root_path, "scaler.joblib")
        self.__read_data__()

    def print_data_stats(self, data, stage):
        """Helper function to print data statistics"""
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
        logger.info(f"\n{stage} statistics: {stats}")
        return stats

    def log_transform(self, data):
        """Apply log transformation with sign preservation"""
        data = np.array(data, dtype=np.float32)
        transformed = np.zeros_like(data)

        # Store signs and work with absolute values
        signs = np.sign(data)
        abs_data = np.abs(data)

        # For values with magnitude > 10
        mask_high = abs_data > 10
        transformed[mask_high] = np.log1p(abs_data[mask_high] - 10) + np.log(10)

        # Keep values between -10 and 10 unchanged
        mask_mid = abs_data <= 10
        transformed[mask_mid] = data[mask_mid]

        # Handle any potential NaN or inf values
        transformed = np.nan_to_num(
            transformed, nan=0.0, posinf=np.log(1e10), neginf=-np.log(1e10)
        )

        return transformed * signs

    def inverse_transform(self, data):
        """Inverse transform both scaling and log transformation"""
        data = np.array(data)

        if self.scale and self.scaler is not None:
            data = self.scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)

        # Inverse log transform
        signs = np.sign(data)
        abs_data = np.abs(data)

        transformed = np.zeros_like(data, dtype=np.float32)

        # Inverse transform for values that were > 10
        mask_high = abs_data > np.log(10)
        transformed[mask_high] = (
            np.exp(abs_data[mask_high] - np.log(10)) + 10
        ) * signs[mask_high]

        # Values <= log(10) remain unchanged
        mask_low = abs_data <= np.log(10)
        transformed[mask_low] = data[mask_low]

        return transformed

    def __read_data__(self):
        try:
            if self.flag == "test":
                data_path = os.path.join(self.root_path, "test_data.pkl")
                with open(data_path, "rb") as f:
                    data_dict = pickle.load(f)

                self.data_x = data_dict["series"]
                self.file_names = data_dict["file_names"]

                # Print initial statistics
                # self.print_data_stats(self.data_x, "Initial test data")

                # Apply log transformation
                for i in range(len(self.data_x)):
                    self.data_x[i] = self.log_transform(self.data_x[i])

                # self.print_data_stats(self.data_x, "After log transform (test)")

                if self.scale:
                    try:
                        self.scaler = joblib.load(self.scaler_path)
                        for i in range(len(self.data_x)):
                            self.data_x[i] = self.scaler.transform(
                                self.data_x[i].reshape(-1, 1)
                            ).reshape(-1)
                            # Clip values to prevent extremes
                            self.data_x[i] = np.clip(self.data_x[i], -10, 10)

                        # self.print_data_stats(self.data_x, "After scaling (test)")
                    except FileNotFoundError:
                        raise ValueError(
                            "Scaler file not found. Please train the model first."
                        )

            else:  # train or val
                split_name = "train" if self.flag == "train" else "val"
                data_path = os.path.join(self.root_path, f"{split_name}_data.pkl")

                with open(data_path, "rb") as f:
                    data_dict = pickle.load(f)

                self.data_x = data_dict["series"]
                self.data_y = data_dict["labels"]
                self.data_y[self.data_y == -1.0] = 0.0  # Convert -1 to 0

                # Print initial statistics
                # self.print_data_stats(self.data_x, f"Initial {split_name} data")

                # Apply log transformation
                for i in range(len(self.data_x)):
                    self.data_x[i] = self.log_transform(self.data_x[i])

                # self.print_data_stats(
                #     self.data_x, f"After log transform ({split_name})"
                # )

                # Apply scaling
                if self.scale:
                    if self.flag == "train":
                        # For training, fit a new scaler
                        self.scaler = RobustScaler(quantile_range=(5.0, 95.0))
                        all_data = np.concatenate(
                            [x.reshape(-1, 1) for x in self.data_x]
                        )
                        self.scaler.fit(all_data)
                        joblib.dump(self.scaler, self.scaler_path)
                    else:
                        try:
                            self.scaler = joblib.load(self.scaler_path)
                        except FileNotFoundError:
                            raise ValueError(
                                "Scaler file not found. Please train the model first."
                            )

                    # Transform the data using the scaler
                    for i in range(len(self.data_x)):
                        self.data_x[i] = self.scaler.transform(
                            self.data_x[i].reshape(-1, 1)
                        ).reshape(-1)
                        # Clip values to prevent extremes
                        self.data_x[i] = np.clip(self.data_x[i], -10, 10)

                    # self.print_data_stats(self.data_x, f"After scaling ({split_name})")

                if self.flag == "train":
                    # Generate shuffle indices
                    shuffle_idx = np.random.permutation(len(self.data_x))
                    self.data_x = [self.data_x[i] for i in shuffle_idx]
                    self.data_y = [self.data_y[i] for i in shuffle_idx]

        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}")
            raise

    def __getitem__(self, index):
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

                # Check for NaN or Inf values before returning
                if np.any(np.isnan(seq_x)) or np.any(np.isinf(seq_x)):
                    logger.warning(f"Found NaN or Inf values in sequence {index}")
                    # Replace NaN/Inf with 0 to prevent training issues
                    seq_x = np.nan_to_num(seq_x, nan=0.0, posinf=10.0, neginf=-10.0)

                return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(
                    seq_y, dtype=torch.float32
                )
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {index}: {str(e)}")
            raise

    def __len__(self):
        return len(self.data_x)

    def get_label_distribution(self):
        """Helper method to get the distribution of labels"""
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


class Dataset_ETT_hour(Dataset):
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

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
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


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
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

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
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
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
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


class Dataset_M4(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="pred",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="15min",
        seasonal_patterns="Yearly",
    ):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == "train":
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [
                v[~np.isnan(v)]
                for v in dataset.values[dataset.groups == self.seasonal_patterns]
            ]
        )  # split different frequencies
        self.ids = np.array(
            [i for i in dataset.ids[dataset.groups == self.seasonal_patterns]]
        )
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(
            low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
            high=len(sampled_timeseries),
            size=1,
        )[0]

        insample_window = sampled_timeseries[
            max(0, cut_point - self.seq_len) : cut_point
        ]
        insample[-len(insample_window) :, 0] = insample_window
        insample_mask[-len(insample_window) :, 0] = 1.0
        outsample_window = sampled_timeseries[
            cut_point
            - self.label_len : min(len(sampled_timeseries), cut_point + self.pred_len)
        ]
        outsample[: len(outsample_window), 0] = outsample_window
        outsample_mask[: len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len :]
            insample[i, -len(ts) :] = ts_last_window
            insample_mask[i, -len(ts) :] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, "train.csv"))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, "test.csv"))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = pd.read_csv(
            os.path.join(root_path, "test_label.csv")
        ).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, "swat_train2.csv"))
        test_data = pd.read_csv(os.path.join(root_path, "swat2.csv"))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.test_labels[0 : self.win_size]
            )
        elif self.flag == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(
            root_path, file_list=file_list, flag=flag
        )
        self.all_IDs = (
            self.all_df.index.unique()
        )  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs), print(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, "*"))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception(
                "No files found using: {}".format(os.path.join(root_path, "*"))
            )
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith(".ts")]
        if len(input_paths) == 0:
            pattern = "*.ts"
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(
            input_paths[0]
        )  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(
            filepath, return_separate_X_and_y=True, replace_missing_vals_with="NaN"
        )
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(
            labels.cat.codes, dtype=np.int8
        )  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)
        ).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if (
            np.sum(horiz_diffs) > 0
        ):  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if (
            np.sum(vert_diffs) > 0
        ):  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat(
            (
                pd.DataFrame({col: df.loc[row, col] for col in df.columns})
                .reset_index(drop=True)
                .set_index(pd.Series(lengths[row, 0] * [row]))
                for row in range(df.shape[0])
            ),
            axis=0,
        )

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if (
            self.root_path.count("EthanolConcentration") > 0
        ):  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(
                torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values

        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(
                batch_x, labels, self.args
            )

            batch_x = batch_x.reshape((1 * seq_len, num_columns))
        return self.instance_norm(torch.from_numpy(batch_x)), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
