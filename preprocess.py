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
    # labels[labels == 0.0] = 0.0
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
