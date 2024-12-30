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

warnings.filterwarnings("ignore")

from skmultilearn.model_selection import iterative_train_test_split
from sklearn.neighbors import NearestNeighbors
import random

import numpy as np
from scipy.interpolate import interp1d
import random


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
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        scaling=False,
        timeenc=0,
        freq="h",
        seed=42,
        val_ratio=0.2,
    ):
        self.args = args
        self.features = features
        self.scale = False  # scaling
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag.lower()
        self.root_path = root_path
        self.seed = seed
        self.val_ratio = val_ratio
        self.kept_indices = [5, 12, 54, 65, 66, 71]  # Indices to keep
        np.random.seed(seed)
        self.scaler = StandardScaler()
        self.__read_data__()

    # def __read_data__(self):
    #     scaler_path = os.path.join(self.root_path, "scaler.save")

    #     if self.flag == "test":
    #         data_path = os.path.join(self.root_path, "test_data.pkl")
    #         with open(data_path, "rb") as f:
    #             data_dict = pickle.load(f)

    #         self.data_x = data_dict["series"]
    #         self.file_names = data_dict["file_names"]

    #         if self.scale and os.path.exists(scaler_path):
    #             self.scaler = joblib.load(scaler_path)
    #             for i in range(len(self.data_x)):
    #                 self.data_x[i] = self.scaler.transform(
    #                     self.data_x[i].reshape(-1, 1)
    #                 ).reshape(-1)

    #     else:  # train or val
    #         split_name = "train" if self.flag == "train" else "val"
    #         data_path = os.path.join(self.root_path, f"{split_name}_data.pkl")

    #         with open(data_path, "rb") as f:
    #             data_dict = pickle.load(f)

    #         self.data_x = data_dict["series"]
    #         self.data_y = data_dict["labels"]
    #         self.data_y[self.data_y == -1.0] = 0.0  # Convert -1 to 0

    #         if self.scale:
    #             if self.flag == "train":
    #                 # Fit scaler on all training data
    #                 all_data = np.concatenate(self.data_x)
    #                 self.scaler.fit(all_data.reshape(-1, 1))
    #                 dump(self.scaler, scaler_path)
    #             elif os.path.exists(scaler_path):
    #                 self.scaler = load(scaler_path)

    #             # Transform each series individually
    #             for i in range(len(self.data_x)):
    #                 self.data_x[i] = self.scaler.transform(
    #                     self.data_x[i].reshape(-1, 1)
    #                 ).reshape(-1)

    def __read_data__(self):
        scaler_path = os.path.join(self.root_path, "scaler.save")

        if self.flag == "test":
            data_path = os.path.join(self.root_path, "test_data.pkl")
            with open(data_path, "rb") as f:
                data_dict = pickle.load(f)

            self.data_x = data_dict["series"]
            self.file_names = data_dict["file_names"]

            if self.scale and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                for i in range(len(self.data_x)):
                    self.data_x[i] = self.scaler.transform(
                        self.data_x[i].reshape(-1, 1)
                    ).reshape(-1)

        else:  # train or val
            split_name = "train" if self.flag == "train" else "val"
            data_path = os.path.join(self.root_path, f"{split_name}_data.pkl")

            with open(data_path, "rb") as f:
                data_dict = pickle.load(f)

            self.data_x = data_dict["series"]
            self.data_y = data_dict["labels"]
            self.data_y[self.data_y == -1.0] = 0.0  # Convert -1 to 0

            # Apply scaling before augmentation if needed
            if self.scale:
                if self.flag == "train":
                    all_data = np.concatenate(self.data_x)
                    self.scaler.fit(all_data.reshape(-1, 1))
                    dump(self.scaler, scaler_path)
                elif os.path.exists(scaler_path):
                    self.scaler = load(scaler_path)

                for i in range(len(self.data_x)):
                    self.data_x[i] = self.scaler.transform(
                        self.data_x[i].reshape(-1, 1)
                    ).reshape(-1)

            # Apply augmentations only for training data
            if self.flag == "train":
                # augmented_x = []
                # augmented_y = []

                # # Generate augmentations for each original sequence
                # for i in range(len(self.data_x)):
                #     # Apply different augmentation methods
                #     aug_sequences, aug_labels = time_series_augmentations(
                #         self.data_x[i],
                #         self.data_y[i],
                #         cut_ratio=0.4,
                #         num_augmentations=1,  # Generate one augmented version per original
                #     )

                #     augmented_x.extend(aug_sequences)
                #     augmented_y.extend(aug_labels)

                # # Combine original and augmented data
                # self.data_x = np.concatenate([self.data_x, augmented_x], axis=0)
                # self.data_y = np.concatenate([self.data_y, augmented_y], axis=0)

                # Shuffle the combined dataset
                shuffle_idx = np.random.permutation(len(self.data_x))
                # print(
                #     "DEBUG",
                #     len(shuffle_idx),
                #     type(shuffle_idx),
                #     type(self.data_x),
                #     type(self.data_y),
                # )

                # Generate shuffle indices
                shuffle_idx = np.random.permutation(len(self.data_x))

                # Shuffle both data_x and data_y using the generated indices
                shuffled_data_x = [self.data_x[i] for i in shuffle_idx]
                shuffled_data_y = [self.data_y[i] for i in shuffle_idx]

                # Convert back to NumPy arrays if needed
                self.data_x = shuffled_data_x
                self.data_y = shuffled_data_y

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        if len(seq_x.shape) == 1:
            seq_x = np.expand_dims(seq_x, axis=1)
        if self.flag == "test":

            return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(
                torch.zeros([1, 94]), dtype=torch.float32
            )
        else:
            seq_y = self.data_y[index]

            # if (
            #     hasattr(self.args, "augmentation_ratio")
            #     and self.args.augmentation_ratio > 0
            # ) or 
            if self.flag == "train":
                seq_x, seq_y, _ = run_augmentation_single(seq_x, seq_y, self.args)
            # print("DEBUG", seq_x.shape)
            return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(
                seq_y, dtype=torch.float32
            )

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) if self.scale else data

    def get_label_distribution(self):
        """Helper method to get the distribution of labels"""
        if hasattr(self, "data_y"):
            return np.sum(self.data_y, axis=0)
        return None


class BalancedDataset_BBB(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        scaling=False,
        timeenc=0,
        freq="h",
        seed=42,
        val_ratio=0.2,
        oversample_threshold=0.1,  # Threshold for minority class identification
    ):
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features = features
        self.scale = scaling
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag.lower()
        self.root_path = root_path
        self.seed = seed
        self.val_ratio = val_ratio
        self.kept_indices = [5, 12, 54, 65, 66, 71]
        self.oversample_threshold = oversample_threshold

        # Define hierarchical relationships (parent -> children)
        self.hierarchy = {
            "Sensor": [
                0,
                1,
                3,
                7,
                8,
                9,
                10,
                11,
                13,
                17,
                18,
                20,
                22,
                24,
                25,
                26,
                28,
                29,
                30,
                31,
                32,
                33,
                35,
                36,
                40,
                41,
                42,
                44,
                48,
                49,
                50,
                52,
                55,
                56,
                57,
                58,
                59,
                60,
                62,
                63,
                69,
                72,
                73,
                75,
                77,
                79,
                81,
                82,
                83,
                84,
                85,
                86,
                88,
                89,
                90,
                91,
                93,
            ],
            "Setpoint": [
                2,
                4,
                14,
                15,
                16,
                19,
                21,
                23,
                27,
                34,
                37,
                38,
                39,
                43,
                45,
                46,
                47,
                51,
                53,
                61,
                64,
                68,
                70,
                74,
                76,
                78,
                80,
                87,
                92,
            ],
        }

        np.random.seed(seed)
        self.__read_data__()
        if self.flag == "train":
            self.__balance_dataset__()

    def __get_parent_label__(self, index):
        """Return parent label index for a given child index"""
        for parent, children in self.hierarchy.items():
            if index in children:
                return (
                    65 if parent == "Sensor" else 66
                )  # Using the kept indices for Sensor and Setpoint
        return None

    def __balance_dataset__(self):
        """Balance the dataset using hierarchical-aware oversampling"""
        if not hasattr(self, "data_y"):
            return

        # Calculate class frequencies
        class_freq = np.sum(self.data_y, axis=0)
        max_freq = np.max(class_freq)

        # Identify minority classes (using threshold)
        minority_mask = class_freq / max_freq < self.oversample_threshold
        minority_indices = np.where(minority_mask)[0]

        # Store original data
        original_x = self.data_x
        original_y = self.data_y

        additional_samples_x = []
        additional_samples_y = []

        # Oversample minority classes
        for idx in minority_indices:
            if idx not in self.kept_indices:
                continue

            # Find samples that have this class
            class_samples = np.where(original_y[:, idx] == 1)[0]
            if len(class_samples) == 0:
                continue

            # Calculate number of samples needed
            target_count = int(max_freq * 0.4)  # Aiming for 40% of majority class
            num_samples = target_count - len(class_samples)

            if num_samples <= 0:
                continue

            # Get parent label index
            parent_idx = self.__get_parent_label__(idx)

            # Sample with replacement
            sampled_indices = np.random.choice(
                class_samples, size=num_samples, replace=True
            )

            for sample_idx in sampled_indices:
                # Add small random noise to features to avoid exact duplicates
                noisy_x = original_x[sample_idx] + np.random.normal(
                    0, 0.01, original_x[sample_idx].shape
                )
                new_y = original_y[sample_idx].copy()

                # Ensure parent label is also set
                if parent_idx is not None:
                    new_y[parent_idx] = 1

                additional_samples_x.append(noisy_x)
                additional_samples_y.append(new_y)

        if additional_samples_x:
            # Combine original and oversampled data
            self.data_x = np.vstack([original_x, np.stack(additional_samples_x)])
            self.data_y = np.vstack([original_y, np.stack(additional_samples_y)])

            # Shuffle the combined dataset
            shuffle_idx = np.random.permutation(len(self.data_x))
            self.data_x = self.data_x[shuffle_idx]
            self.data_y = self.data_y[shuffle_idx]

    def __read_data__(self):
        scaler_path = os.path.join(self.root_path, "scaler.save")
        self.scaler = StandardScaler()

        if self.flag == "test":
            X_test_path = os.path.join(self.root_path, "X_test.npy")
            self.data_x = np.load(X_test_path)
            self.data_x = np.transpose(self.data_x, (0, 2, 1))

            file_names_path = os.path.join(self.root_path, "test_file_names.txt")
            with open(file_names_path, "r") as f:
                self.file_names = [line.strip() for line in f.readlines()]

            if self.scale:
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    self.data_x = self.scaler.transform(
                        self.data_x.reshape(-1, self.data_x.shape[-1])
                    ).reshape(self.data_x.shape)
                else:
                    raise FileNotFoundError("Scaler file not found.")

        elif self.flag == "val":
            X_val_path = os.path.join(self.root_path, "X_val.npy")
            y_val_path = os.path.join(self.root_path, "y_val.npy")

            self.data_x = np.load(X_val_path)
            self.data_x = np.transpose(self.data_x, (0, 2, 1))
            self.data_y = np.load(y_val_path)

            # Zero out non-kept indices
            mask = np.zeros(self.data_y.shape[-1], dtype=bool)  # * -1
            mask[self.kept_indices] = True
            self.data_y[self.data_y == -1.0] = 0.0
            # self.data_y = self.data_y * mask

            if self.scale:
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    self.data_x = self.scaler.transform(
                        self.data_x.reshape(-1, self.data_x.shape[-1])
                    ).reshape(self.data_x.shape)
                else:
                    raise FileNotFoundError(
                        "Scaler file not found. Train the model first."
                    )

        else:  # train
            X_path = os.path.join(self.root_path, "X_train.npy")
            y_path = os.path.join(self.root_path, "y_train.npy")

            self.data_x = np.load(X_path)
            self.data_x = np.transpose(self.data_x, (0, 2, 1))
            self.data_y = np.load(y_path)
            # Fix: make -1's as 0's for train
            self.data_y[self.data_y == -1.0] = 0.0

            # Zero out non-kept indices
            mask = np.zeros(self.data_y.shape[-1], dtype=bool)
            mask[self.kept_indices] = True
            # self.data_y = self.data_y * mask

            if self.scale:
                self.scaler.fit(self.data_x.reshape(-1, self.data_x.shape[-1]))
                joblib.dump(self.scaler, scaler_path)
                self.data_x = self.scaler.transform(
                    self.data_x.reshape(-1, self.data_x.shape[-1])
                ).reshape(self.data_x.shape)

    def __getitem__(self, index):
        if self.flag == "test":
            seq_x = self.data_x[index]
            return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(
                torch.zeros([1, 94]), dtype=torch.float32
            )
        else:
            seq_x = self.data_x[index]
            seq_y = self.data_y[index]
            return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(
                seq_y, dtype=torch.float32
            )

    def __len__(self):
        if self.flag == "test":
            return len(self.data_x)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_label_distribution(self):
        """Helper method to get the distribution of labels"""
        if hasattr(self, "data_y"):
            return np.sum(self.data_y, axis=0)
        return None


# class Dataset_BBB(Dataset):
#     def __init__(
#         self,
#         args,
#         root_path,
#         flag="train",
#         size=None,
#         features="S",
#         scale=True,
#         timeenc=0,
#         freq="h",
#         seed=42
#     ):
#         self.args = args
#         if size is None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]

#         self.features = features
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.flag = flag.lower()
#         self.root_path = root_path
#         self.seed = seed
#         self.__read_data__()

#     def __read_data__(self):
#         scaler_path = os.path.join(self.root_path, "scaler.save")
#         self.scaler = StandardScaler()

#         if self.flag == "test":
#             # Load test data
#             X_test_path = os.path.join(self.root_path, "X_test.npy")
#             self.data_x = np.load(X_test_path)
#             self.data_x = np.transpose(self.data_x, (0, 2, 1))

#             # Load file names for reference
#             file_names_path = os.path.join(self.root_path, "test_file_names.txt")
#             with open(file_names_path, 'r') as f:
#                 self.file_names = [line.strip() for line in f.readlines()]

#             if os.path.exists(scaler_path):
#                 self.scaler = joblib.load(scaler_path)
#                 self.data_x = self.scaler.transform(
#                     self.data_x.reshape(-1, self.data_x.shape[-1])
#                 ).reshape(self.data_x.shape)
#             else:
#                 raise FileNotFoundError("Scaler file not found.")

#         else:
#             # Load training data
#             X_path = os.path.join(self.root_path, "X_train.npy")
#             y_path = os.path.join(self.root_path, "y_train.npy")

#             self.data_x = np.load(X_path)
#             self.data_x = np.transpose(self.data_x, (0, 2, 1))
#             self.data_y = np.load(y_path)

#             X_train, y_train, X_val, y_val = iterative_train_test_split(
#                 self.data_x, self.data_y, test_size=0.199,
#             )

#             if self.flag == "train":
#                 self.data_x, self.data_y = X_train, y_train
#             else:  # val
#                 self.data_x, self.data_y = X_val, y_val

#             if self.scale:
#                 train_data = X_train
#                 self.scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
#                 joblib.dump(self.scaler, scaler_path)
#                 self.data_x = self.scaler.transform(
#                     self.data_x.reshape(-1, self.data_x.shape[-1])
#                 ).reshape(self.data_x.shape)

#     def __getitem__(self, index):
#         if self.flag == "test":
#             seq_x = self.data_x[index]
#             return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(torch.zeros([1,94]), dtype=torch.float32)
#         else:
#             seq_x = self.data_x[index]
#             seq_y = self.data_y[index]
#             return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

#     def __len__(self):
#         if self.flag == "test":
#             return len(self.data_x)
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

#     def get_label_distribution(self):
#         """Helper method to get the distribution of labels"""
#         if hasattr(self, 'data_y'):
#             return np.sum(self.data_y, axis=0)
#         return None


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

        # print("Batch X shape:", batch_x.shape)
        # print("Labels shape:", labels.shape)
        # import sys

        # sys.exit()
        return self.instance_norm(torch.from_numpy(batch_x)), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


