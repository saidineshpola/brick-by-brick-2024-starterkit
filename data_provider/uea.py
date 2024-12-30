import os
import numpy as np
import pandas as pd
import torch


def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input.
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """
    batch_size = len(data)
    features, labels = zip(*data)

    # Get original sequence lengths
    lengths = [X.shape[0] for X in features]
    if max_len is None:
        max_len = max(lengths)

    # Initialize tensors with zeros
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])
    padding_masks = torch.zeros((batch_size, max_len), dtype=torch.bool)  # Initialize with False

    # Fill in the actual data and create padding masks
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        padding_masks[i, :end] = True  # Set True for actual data, False for padding

    targets = torch.stack(labels, dim=0)

    return X, targets, padding_masks

def create_padding_mask(X):
    """
    Creates a padding mask by detecting zero vectors from the end of sequences.
    Args:
        X: Input tensor of shape (batch_size, seq_length, feat_dim)
    Returns:
        Boolean tensor of shape (batch_size, seq_length) where True indicates non-padding positions
    """
    # Calculate L2 norm across feature dimension
    norms = torch.norm(X, dim=2)  # Shape: (batch_size, seq_length)

    # Create a mask where True indicates non-zero vectors
    masks = norms > 0

    # For each sequence, flip all masks to False once we encounter the first False
    # when moving from right to left
    batch_size, seq_length = masks.shape
    for i in range(batch_size):
        # Find the last non-zero position
        last_nonzero = -1
        for j in range(seq_length - 1, -1, -1):
            if masks[i, j]:
                last_nonzero = j
                break
        # Set all positions after last_nonzero to False
        if last_nonzero != -1:
            masks[i, last_nonzero + 1 :] = False

    return masks


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(
        self,
        norm_type="standardization",
        mean=None,
        std=None,
        min_val=None,
        max_val=None,
    ):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method="linear", limit_direction="both")
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y
