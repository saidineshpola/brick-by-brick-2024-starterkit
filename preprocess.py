import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
import pandas as pd


def create_validation_set(df, val_percentage=0.2, hierarchy=None):
    """
    Create validation set while respecting hierarchical relationships

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing file paths and labels
    val_percentage : float
        Percentage of data to use for validation (default: 0.2)
    hierarchy : list
        List of column names defining the hierarchy for splitting

    Returns:
    --------
    train_df, val_df : tuple of pandas.DataFrame
        Split DataFrames for training and validation
    """
    if hierarchy is None or len(hierarchy) == 0:
        # Simple random split if no hierarchy is provided
        shuffled_df = df.sample(frac=1, random_state=42)
        split_idx = int(len(df) * (1 - val_percentage))
        return shuffled_df.iloc[:split_idx], shuffled_df.iloc[split_idx:]

    # Get unique combinations of hierarchy columns
    hierarchy_groups = df[hierarchy].drop_duplicates()

    # Random split at the hierarchy level
    val_size = int(len(hierarchy_groups) * val_percentage)
    val_groups = hierarchy_groups.sample(n=val_size, random_state=42)

    # Create masks for splitting
    val_mask = df[hierarchy].merge(val_groups, how="inner", on=hierarchy).index
    val_df = df.loc[val_mask]
    train_df = df.drop(val_mask)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    return train_df, val_df


def get_percentile_length(directory, percentile=95):
    """
    Determine time series length statistics from a directory of files

    Parameters:
    -----------
    directory : str
        Directory containing pickle files
    percentile : int, optional (default=95)
        Percentile of time series lengths to calculate

    Returns:
    --------
    dict
        Dictionary containing length statistics
    """
    lengths = []
    files = [f for f in os.listdir(directory) if f.endswith(".pkl")]

    for file_name in tqdm(files, desc="Checking time series lengths"):
        file_path = os.path.join(directory, file_name)

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            if "v" in data:
                lengths.append(len(data["v"]))
            else:
                print(f"Skipping {file_name}: Unexpected dictionary format")
        except Exception as e:
            print(f"Error checking length for {file_name}: {e}")

    stats = {
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": np.mean(lengths),
        f"{percentile}th_percentile": int(np.percentile(lengths, percentile)),
    }

    print(f"\nLength Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    return stats


def clip_values(data):
    """
    Clip values to range [-1e8, 1e8]

    Parameters:
    -----------
    data : numpy.ndarray
        Input data array

    Returns:
    --------
    numpy.ndarray
        Clipped data
    """
    return np.clip(data, -1e8, 1e8)


def prepare_train_timeseries_dataset(
    df, data_directory, output_directory, split_name="train"
):
    """
    Prepare training/validation time series dataset from pickle files with multi-label support,
    preserving original lengths

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing file paths and labels
    data_directory : str
        Directory containing the pickle files
    output_directory : str
        Directory to save processed data
    split_name : str
        Name of the split ("train" or "val") for saving
    """
    filepath_column = df.columns[0]
    label_columns = df.columns[1:]

    series_data = []
    labels = []
    lengths = []
    file_names = []

    total_files = len(df)
    print(f"Total {split_name} files to process: {total_files}")
    print(f"Label columns: {list(label_columns)}")

    print(f"Loading and processing {split_name} time series data...")
    for _, row in tqdm(df.iterrows(), total=total_files):
        file_name = row[filepath_column]
        file_path = os.path.join(data_directory, file_name)

        try:
            with open(file_path, "rb") as f:
                train_X = pickle.load(f)

            if not ("t" in train_X and "v" in train_X):
                print(f"Skipping {file_name}: Unexpected dictionary format")
                continue

            series_values = np.array(train_X["v"])
            series_values = clip_values(series_values)

            # Store the original series without padding
            series_data.append(series_values.astype(np.float32))
            lengths.append(len(series_values))
            file_names.append(file_name)

            # Store labels
            sample_labels = row[label_columns].values
            labels.append(sample_labels)

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # Convert labels and handle -1 values
    labels = np.array(labels, dtype=float)
    print(f"Before, number of 1s,0s: {np.sum(labels == 1.0)},{np.sum(labels == 0.0)}")
    labels[labels == 0.0] = 0.0
    labels[labels == -1.0] = 0.0
    print(f"Final number of 1s,0s: {np.sum(labels == 1.0)},{np.sum(labels == 0.0)}")
    print(f"Final Shape: {labels.shape}")

    # Save the data
    os.makedirs(output_directory, exist_ok=True)

    # Save as a dictionary to preserve variable lengths
    data = {
        "series": series_data,
        "labels": labels,
        "lengths": np.array(lengths),
        "file_names": file_names,
        "label_columns": list(label_columns),
    }

    with open(os.path.join(output_directory, f"{split_name}_data.pkl"), "wb") as f:
        pickle.dump(data, f)

    metadata = {
        "lengths": np.array(lengths),
        "label_columns": list(label_columns),
        "value_range": {"min": -1e8, "max": 1e8},
    }

    return data, metadata


def prepare_test_timeseries_dataset(test_directory, output_directory):
    """
    Prepare test time series dataset from pickle files without labels,
    preserving original lengths
    """
    series_data = []
    lengths = []
    file_names = []

    test_files = [f for f in os.listdir(test_directory) if f.endswith(".pkl")]
    total_files = len(test_files)
    print(f"Total test files to process: {total_files}")

    print("Loading and processing test time series data...")
    for file_name in tqdm(test_files):
        file_path = os.path.join(test_directory, file_name)

        try:
            with open(file_path, "rb") as f:
                test_X = pickle.load(f)

            if not ("t" in test_X and "v" in test_X):
                print(f"Skipping {file_name}: Unexpected dictionary format")
                continue

            series_values = np.array(test_X["v"])
            series_values = clip_values(series_values)

            # Store the original series without padding
            series_data.append(series_values.astype(np.float32))
            lengths.append(len(series_values))
            file_names.append(file_name)

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # Save the data
    os.makedirs(output_directory, exist_ok=True)

    # Save as a dictionary to preserve variable lengths
    test_data = {
        "series": series_data,
        "lengths": np.array(lengths),
        "file_names": file_names,
    }

    with open(os.path.join(output_directory, "test_data.pkl"), "wb") as f:
        pickle.dump(test_data, f)

    # Save file names separately for convenience
    with open(os.path.join(output_directory, "test_file_names.txt"), "w") as f:
        for name in file_names:
            f.write(f"{name}\n")

    print("\nTest Dataset Information:")
    print(f"Number of series: {len(series_data)}")
    print(f"Length range: {min(lengths)} to {max(lengths)}")

    metadata = {
        "lengths": np.array(lengths),
        "file_names": file_names,
        "value_range": {"min": -1e8, "max": 1e8},
    }

    return test_data, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Process time series data for train and test sets."
    )
    parser.add_argument(
        "--train_directory",
        type=str,
        default="./train_X/",
        help="Directory containing training pickle files",
    )
    parser.add_argument(
        "--test_directory",
        type=str,
        default="./test_X/",
        help="Directory containing test pickle files",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="./train_y_v0.1.0.csv",
        help="CSV file with file paths and labels for training data",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data/BBB/",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=96,
        help="Percentile for length statistics",
    )
    parser.add_argument(
        "--val_percentage",
        type=float,
        default=0.2,
        help="Percentage of data to use for validation",
    )
    parser.add_argument(
        "--hierarchy",
        type=str,
        nargs="+",
        default=None,
        help="List of column names defining the hierarchy for splitting",
    )

    args = parser.parse_args()

    # Get length statistics
    train_stats = get_percentile_length(
        args.train_directory, percentile=args.percentile
    )
    test_stats = get_percentile_length(args.test_directory, percentile=args.percentile)

    # Load and split training data
    trainy_df = pd.read_csv(args.csv_file)
    train_df, val_df = create_validation_set(
        trainy_df, val_percentage=args.val_percentage, hierarchy=args.hierarchy
    )

    # Process training data
    train_data, train_metadata = prepare_train_timeseries_dataset(
        train_df, args.train_directory, args.output_directory, split_name="train"
    )
    print("\nTraining data processing completed")

    # Process validation data
    val_data, val_metadata = prepare_train_timeseries_dataset(
        val_df, args.train_directory, args.output_directory, split_name="val"
    )
    print("\nValidation data processing completed")

    # Process test data
    test_data, test_metadata = prepare_test_timeseries_dataset(
        args.test_directory, args.output_directory
    )
    print("\nTest data processing completed")


if __name__ == "__main__":
    main()
