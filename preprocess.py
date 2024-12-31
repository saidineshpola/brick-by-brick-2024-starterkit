import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

np.random.seed(2024)
labels_to_indx = {
    "Active_Power_Sensor": 0,
    "Air_Flow_Sensor": 1,
    "Air_Flow_Setpoint": 2,
    "Air_Temperature_Sensor": 3,
    "Air_Temperature_Setpoint": 4,
    "Alarm": 5,
    "Angle_Sensor": 6,
    "Average_Zone_Air_Temperature_Sensor": 7,
    "Chilled_Water_Differential_Temperature_Sensor": 8,
    "Chilled_Water_Return_Temperature_Sensor": 9,
    "Chilled_Water_Supply_Flow_Sensor": 10,
    "Chilled_Water_Supply_Temperature_Sensor": 11,
    "Command": 12,
    "Cooling_Demand_Sensor": 13,
    "Cooling_Demand_Setpoint": 14,
    "Cooling_Supply_Air_Temperature_Deadband_Setpoint": 15,
    "Cooling_Temperature_Setpoint": 16,
    "Current_Sensor": 17,
    "Damper_Position_Sensor": 18,
    "Damper_Position_Setpoint": 19,
    "Demand_Sensor": 20,
    "Dew_Point_Setpoint": 21,
    "Differential_Pressure_Sensor": 22,
    "Differential_Pressure_Setpoint": 23,
    "Differential_Supply_Return_Water_Temperature_Sensor": 24,
    "Discharge_Air_Dewpoint_Sensor": 25,
    "Discharge_Air_Temperature_Sensor": 26,
    "Discharge_Air_Temperature_Setpoint": 27,
    "Discharge_Water_Temperature_Sensor": 28,
    "Duration_Sensor": 29,
    "Electrical_Power_Sensor": 30,
    "Energy_Usage_Sensor": 31,
    "Filter_Differential_Pressure_Sensor": 32,
    "Flow_Sensor": 33,
    "Flow_Setpoint": 34,
    "Frequency_Sensor": 35,
    "Heating_Demand_Sensor": 36,
    "Heating_Demand_Setpoint": 37,
    "Heating_Supply_Air_Temperature_Deadband_Setpoint": 38,
    "Heating_Temperature_Setpoint": 39,
    "Hot_Water_Flow_Sensor": 40,
    "Hot_Water_Return_Temperature_Sensor": 41,
    "Hot_Water_Supply_Temperature_Sensor": 42,
    "Humidity_Setpoint": 43,
    "Load_Current_Sensor": 44,
    "Low_Outside_Air_Temperature_Enable_Setpoint": 45,
    "Max_Air_Temperature_Setpoint": 46,
    "Min_Air_Temperature_Setpoint": 47,
    "Outside_Air_CO2_Sensor": 48,
    "Outside_Air_Enthalpy_Sensor": 49,
    "Outside_Air_Humidity_Sensor": 50,
    "Outside_Air_Lockout_Temperature_Setpoint": 51,
    "Outside_Air_Temperature_Sensor": 52,
    "Outside_Air_Temperature_Setpoint": 53,
    "Parameter": 54,
    "Peak_Power_Demand_Sensor": 55,
    "Position_Sensor": 56,
    "Power_Sensor": 57,
    "Pressure_Sensor": 58,
    "Rain_Sensor": 59,
    "Reactive_Power_Sensor": 60,
    "Reset_Setpoint": 61,
    "Return_Air_Temperature_Sensor": 62,
    "Return_Water_Temperature_Sensor": 63,
    "Room_Air_Temperature_Setpoint": 64,
    "Sensor": 65,
    "Setpoint": 66,
    "Solar_Radiance_Sensor": 67,
    "Speed_Setpoint": 68,
    "Static_Pressure_Sensor": 69,
    "Static_Pressure_Setpoint": 70,
    "Status": 71,
    "Supply_Air_Humidity_Sensor": 72,
    "Supply_Air_Static_Pressure_Sensor": 73,
    "Supply_Air_Static_Pressure_Setpoint": 74,
    "Supply_Air_Temperature_Sensor": 75,
    "Supply_Air_Temperature_Setpoint": 76,
    "Temperature_Sensor": 77,
    "Temperature_Setpoint": 78,
    "Thermal_Power_Sensor": 79,
    "Time_Setpoint": 80,
    "Usage_Sensor": 81,
    "Valve_Position_Sensor": 82,
    "Voltage_Sensor": 83,
    "Warmest_Zone_Air_Temperature_Sensor": 84,
    "Water_Flow_Sensor": 85,
    "Water_Temperature_Sensor": 86,
    "Water_Temperature_Setpoint": 87,
    "Wind_Direction_Sensor": 88,
    "Wind_Speed_Sensor": 89,
    "Zone_Air_Dewpoint_Sensor": 90,
    "Zone_Air_Humidity_Sensor": 91,
    "Zone_Air_Humidity_Setpoint": 92,
    "Zone_Air_Temperature_Sensor": 93,
}

hierarchy = {
    "Sensor": [
        "Active_Power_Sensor",
        "Air_Flow_Sensor",
        "Air_Temperature_Sensor",
        "Angle_Sensor",
        "Average_Zone_Air_Temperature_Sensor",
        "Chilled_Water_Differential_Temperature_Sensor",
        "Chilled_Water_Return_Temperature_Sensor",
        "Chilled_Water_Supply_Flow_Sensor",
        "Chilled_Water_Supply_Temperature_Sensor",
        "Cooling_Demand_Sensor",
        "Current_Sensor",
        "Damper_Position_Sensor",
        "Demand_Sensor",
        "Differential_Pressure_Sensor",
        "Differential_Supply_Return_Water_Temperature_Sensor",
        "Discharge_Air_Dewpoint_Sensor",
        "Discharge_Air_Temperature_Sensor",
        "Discharge_Water_Temperature_Sensor",
        "Duration_Sensor",
        "Electrical_Power_Sensor",
        "Energy_Usage_Sensor",
        "Filter_Differential_Pressure_Sensor",
        "Flow_Sensor",
        "Frequency_Sensor",
        "Heating_Demand_Sensor",
        "Hot_Water_Flow_Sensor",
        "Hot_Water_Return_Temperature_Sensor",
        "Hot_Water_Supply_Temperature_Sensor",
        "Load_Current_Sensor",
        "Outside_Air_CO2_Sensor",
        "Outside_Air_Enthalpy_Sensor",
        "Outside_Air_Humidity_Sensor",
        "Outside_Air_Temperature_Sensor",
        "Peak_Power_Demand_Sensor",
        "Position_Sensor",
        "Power_Sensor",
        "Pressure_Sensor",
        "Rain_Sensor",
        "Reactive_Power_Sensor",
        "Return_Air_Temperature_Sensor",
        "Return_Water_Temperature_Sensor",
        "Solar_Radiance_Sensor",
        "Static_Pressure_Sensor",
        "Supply_Air_Humidity_Sensor",
        "Supply_Air_Static_Pressure_Sensor",
        "Supply_Air_Temperature_Sensor",
        "Temperature_Sensor",
        "Thermal_Power_Sensor",
        "Usage_Sensor",
        "Valve_Position_Sensor",
        "Voltage_Sensor",
        "Warmest_Zone_Air_Temperature_Sensor",
        "Water_Flow_Sensor",
        "Water_Temperature_Sensor",
        "Wind_Direction_Sensor",
        "Wind_Speed_Sensor",
        "Zone_Air_Dewpoint_Sensor",
        "Zone_Air_Humidity_Sensor",
        "Zone_Air_Temperature_Sensor",
    ],
    "Setpoint": [
        "Air_Flow_Setpoint",
        "Air_Temperature_Setpoint",
        "Cooling_Demand_Setpoint",
        "Cooling_Supply_Air_Temperature_Deadband_Setpoint",
        "Cooling_Temperature_Setpoint",
        "Damper_Position_Setpoint",
        "Dew_Point_Setpoint",
        "Differential_Pressure_Setpoint",
        "Discharge_Air_Temperature_Setpoint",
        "Flow_Setpoint",
        "Heating_Demand_Setpoint",
        "Heating_Supply_Air_Temperature_Deadband_Setpoint",
        "Heating_Temperature_Setpoint",
        "Humidity_Setpoint",
        "Low_Outside_Air_Temperature_Enable_Setpoint",
        "Max_Air_Temperature_Setpoint",
        "Min_Air_Temperature_Setpoint",
        "Outside_Air_Lockout_Temperature_Setpoint",
        "Outside_Air_Temperature_Setpoint",
        "Reset_Setpoint",
        "Room_Air_Temperature_Setpoint",
        "Speed_Setpoint",
        "Static_Pressure_Setpoint",
        "Supply_Air_Static_Pressure_Setpoint",
        "Supply_Air_Temperature_Setpoint",
        "Temperature_Setpoint",
        "Time_Setpoint",
        "Water_Temperature_Setpoint",
        "Zone_Air_Humidity_Setpoint",
    ],
    "Demand_Sensor": [
        "Cooling_Demand_Sensor",
        "Heating_Demand_Sensor",
        "Peak_Power_Demand_Sensor",
    ],
    "Flow_Sensor": [
        "Air_Flow_Sensor",
        "Chilled_Water_Supply_Flow_Sensor",
        "Hot_Water_Flow_Sensor",
        "Water_Flow_Sensor",
    ],
    "Flow_Setpoint": ["Air_Flow_Setpoint"],
    "Humidity_Setpoint": ["Zone_Air_Humidity_Setpoint"],
    "Power_Sensor": [
        "Active_Power_Sensor",
        "Electrical_Power_Sensor",
        "Reactive_Power_Sensor",
        "Thermal_Power_Sensor",
    ],
    "Pressure_Sensor": [
        "Differential_Pressure_Sensor",
        "Filter_Differential_Pressure_Sensor",
        "Static_Pressure_Sensor",
        "Supply_Air_Static_Pressure_Sensor",
    ],
    "Position_Sensor": [
        "Damper_Position_Sensor",
        "Valve_Position_Sensor",
    ],
    "Temperature_Sensor": [
        "Air_Temperature_Sensor",
        "Average_Zone_Air_Temperature_Sensor",
        "Chilled_Water_Differential_Temperature_Sensor",
        "Chilled_Water_Return_Temperature_Sensor",
        "Chilled_Water_Supply_Temperature_Sensor",
        "Differential_Supply_Return_Water_Temperature_Sensor",
        "Discharge_Air_Temperature_Sensor",
        "Discharge_Water_Temperature_Sensor",
        "Hot_Water_Return_Temperature_Sensor",
        "Hot_Water_Supply_Temperature_Sensor",
        "Outside_Air_Temperature_Sensor",
        "Return_Air_Temperature_Sensor",
        "Return_Water_Temperature_Sensor",
        "Supply_Air_Temperature_Sensor",
        "Warmest_Zone_Air_Temperature_Sensor",
        "Water_Temperature_Sensor",
        "Zone_Air_Temperature_Sensor",
    ],
    "Temperature_Setpoint": [
        "Air_Temperature_Setpoint",
        "Cooling_Temperature_Setpoint",
        "Discharge_Air_Temperature_Setpoint",
        "Heating_Temperature_Setpoint",
        "Max_Air_Temperature_Setpoint",
        "Min_Air_Temperature_Setpoint",
        "Outside_Air_Lockout_Temperature_Setpoint",
        "Outside_Air_Temperature_Setpoint",
        "Room_Air_Temperature_Setpoint",
        "Supply_Air_Temperature_Setpoint",
        "Water_Temperature_Setpoint",
    ],
    "Usage_Sensor": ["Energy_Usage_Sensor"],
    "Air_Temperature_Sensor": [
        "Average_Zone_Air_Temperature_Sensor",
        "Discharge_Air_Temperature_Sensor",
        "Outside_Air_Temperature_Sensor",
        "Return_Air_Temperature_Sensor",
        "Supply_Air_Temperature_Sensor",
        "Warmest_Zone_Air_Temperature_Sensor",
        "Zone_Air_Temperature_Sensor",
    ],
    "Air_Temperature_Setpoint": [
        "Discharge_Air_Temperature_Setpoint",
        "Max_Air_Temperature_Setpoint",
        "Min_Air_Temperature_Setpoint",
        "Outside_Air_Temperature_Setpoint",
        "Room_Air_Temperature_Setpoint",
        "Supply_Air_Temperature_Setpoint",
    ],
    "Differential_Pressure_Sensor": [
        "Filter_Differential_Pressure_Sensor",
    ],
    "Static_Pressure_Sensor": [
        "Supply_Air_Static_Pressure_Sensor",
    ],
    "Static_Pressure_Setpoint": [
        "Supply_Air_Static_Pressure_Setpoint",
    ],
    "Water_Temperature_Sensor": [
        "Differential_Supply_Return_Water_Temperature_Sensor",
        "Discharge_Water_Temperature_Sensor",
        "Return_Water_Temperature_Sensor",
    ],
}


def create_validation_set(trainy_df, val_percentage=0.2, hierarchy=None):
    """
    Create validation set while respecting hierarchical relationships

    Parameters:
    -----------
    trainy_df : pandas.DataFrame
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
        shuffled_df = trainy_df.sample(frac=1, random_state=42)
        split_idx = int(len(trainy_df) * (1 - val_percentage))
        return shuffled_df.iloc[:split_idx], shuffled_df.iloc[split_idx:]

    print("Using Hierarchy Split")
    # Calculate the number of samples needed for validation
    total_samples = len(trainy_df)
    val_size = int(total_samples * val_percentage)

    # Initialize lists to hold indices for train and validation sets
    val_indices = set()
    train_indices = set(range(total_samples))

    # Ensure at least 20% of each child class is included in the validation set
    for parent, children in hierarchy.items():
        # Only Check for 3 level hierarchy
        if len(parent.split("_")) != 3:
            continue
        for child in children:
            child_indices = trainy_df.index[trainy_df[child] == 1].tolist()
            if not child_indices:
                continue
            # Calculate the number of samples needed for this child class
            child_val_size = max(1, int(len(child_indices) * val_percentage))
            # Randomly select indices for validation
            selected_val_indices = np.random.choice(
                child_indices, child_val_size, replace=False
            )
            val_indices.update(selected_val_indices)

    # Add at least 1 entry to validation for classes with 0 entries in val but more than 4 in train
    for col in trainy_df.columns:
        if (trainy_df[col] == 1).sum() >= 5 and (
            trainy_df.iloc[list(val_indices)][col] == 1
        ).sum() == 0:
            additional_index = trainy_df.index[
                (trainy_df[col] == 1) & (trainy_df.index.isin(train_indices))
            ].tolist()
            if additional_index:
                val_indices.add(additional_index[0])

    # Add remaining samples to reach the desired validation size
    remaining_val_size = val_size - len(val_indices)
    if remaining_val_size > 0:
        remaining_indices = list(train_indices - val_indices)
        additional_val_indices = np.random.choice(
            remaining_indices, remaining_val_size, replace=False
        )
        val_indices.update(additional_val_indices)

    # Convert sets to lists
    val_indices = list(val_indices)
    train_indices = list(train_indices - set(val_indices))

    # Ensure training set is larger than validation set
    if len(train_indices) < len(val_indices):
        excess_val_indices = val_indices[len(train_indices) :]
        val_indices = val_indices[: len(train_indices)]
        train_indices.extend(excess_val_indices)

    # Split the data
    train_df = trainy_df.iloc[train_indices]
    val_df = trainy_df.iloc[val_indices]

    numerical_cols = train_df.select_dtypes(include=["float64", "int64"]).columns

    # Iterate over each numerical column and count occurrences of 1.0
    for col in numerical_cols:
        count_ones = (val_df[col] == 1.0).sum()
        count_ones_train = (train_df[col] == 1.0).sum()
        print(f"Count of Column '{col}' : train {count_ones_train}, val {count_ones} ")

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


def resample_timeseries(times, values, interval="10min"):
    """
    Resample time series data to a specified interval

    Parameters:
    -----------
    times : array-like
        Array of timestamps in nanoseconds
    values : array-like
        Array of corresponding values
    interval : str, optional (default='10min')
        Resampling interval in pandas offset string format

    Returns:
    --------
    tuple
        Resampled times (as datetime), resampled values
    """
    # Convert nanosecond timestamps to pandas datetime
    df = pd.DataFrame({"time": pd.to_datetime(times, unit="ns"), "value": values})
    df.set_index("time", inplace=True)

    # Resample data
    resampled = df.resample(interval).mean().interpolate(method="linear")

    return resampled.index.values, resampled["value"].values


def prepare_train_timeseries_dataset(
    df, data_directory, output_directory, split_name="train"
):
    """
    Prepare training/validation time series dataset from pickle files with multi-label support,
    including resampling to 10-minute intervals
    """
    filepath_column = df.columns[0]
    label_columns = df.columns[1:]

    series_data = []
    labels = []
    lengths = []
    file_names = []

    total_files = len(df)
    print(f"Total {split_name} files to process: {total_files}")

    print(f"Loading and processing {split_name} time series data...")
    for _, row in tqdm(df.iterrows(), total=total_files):
        file_name = row[filepath_column]
        file_path = os.path.join(data_directory, file_name)

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            if not ("t" in data and "v" in data):
                print(f"Skipping {file_name}: Unexpected dictionary format")
                continue

            # Resample the time series
            resampled_times, resampled_values = resample_timeseries(
                data["t"], data["v"]
            )
            series_values = np.clip(resampled_values, -1e9, 1e9)

            # Store the resampled series
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
    labels[labels == -1.0] = 0.0
    print(f"Final number of 1s,0s: {np.sum(labels == 1.0)},{np.sum(labels == 0.0)}")
    print(f"Final Shape: {labels.shape}")

    # Save the data
    os.makedirs(output_directory, exist_ok=True)

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
        "sampling_interval": "10min",
    }

    return data, metadata


def prepare_test_timeseries_dataset(test_directory, output_directory):
    """
    Prepare test time series dataset from pickle files without labels,
    including resampling to 10-minute intervals
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
                data = pickle.load(f)

            if not ("t" in data and "v" in data):
                print(f"Skipping {file_name}: Unexpected dictionary format")
                continue

            # Resample the time series
            resampled_times, resampled_values = resample_timeseries(
                data["t"], data["v"]
            )
            series_values = np.clip(resampled_values, -1e9, 1e9)

            # Store the resampled series
            series_data.append(series_values.astype(np.float32))
            lengths.append(len(series_values))
            file_names.append(file_name)

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # Save the data
    os.makedirs(output_directory, exist_ok=True)

    test_data = {
        "series": series_data,
        "lengths": np.array(lengths),
        "file_names": file_names,
    }

    with open(os.path.join(output_directory, "test_data.pkl"), "wb") as f:
        pickle.dump(test_data, f)

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
        "sampling_interval": "10min",
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

    args = parser.parse_args()

    # Get length statistics
    train_stats = get_percentile_length(
        args.train_directory, percentile=args.percentile
    )
    test_stats = get_percentile_length(args.test_directory, percentile=args.percentile)

    # Load and split training data
    global hierarchy
    trainy_df = pd.read_csv(args.csv_file)
    train_df, val_df = create_validation_set(
        trainy_df, val_percentage=args.val_percentage, hierarchy=hierarchy
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
