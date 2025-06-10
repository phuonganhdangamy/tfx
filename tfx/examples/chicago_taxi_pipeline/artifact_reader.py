#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import display_util
from tensorflow_metadata.proto.v0 import statistics_pb2
from tensorflow_metadata.proto.v0 import anomalies_pb2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""
Script to read and visualize TFX pipeline artifacts
"""

def read_feature_stats(pb_file_path):
    """Read FeatureStats.pb file"""
    print(f"Reading FeatureStats from: {pb_file_path}")

    # Load the statistics
    stats = tfdv.load_stats_binary(pb_file_path)

    # Display basic info
    print(f"Number of datasets: {len(stats.datasets)}")
    for i, dataset in enumerate(stats.datasets):
        print(f"\nDataset {i}:")
        print(f"  Name: {dataset.name}")
        print(f"  Number of examples: {dataset.num_examples}")
        print(f"  Number of features: {len(dataset.features)}")

        # Show first few features
        print("\nFeature statistics:")
        for j, feature in enumerate(dataset.features[:5]):  # First 5 features
            print(f"  {j+1}. {feature.name}")
            if feature.HasField('num_stats'):
                print(f"     Type: Numeric")
                print(f"     Min: {feature.num_stats.min}")
                print(f"     Max: {feature.num_stats.max}")
                print(f"     Mean: {feature.num_stats.mean}")
            elif feature.HasField('string_stats'):
                print(f"     Type: String")
                print(f"     Unique values: {feature.string_stats.unique}")

        if len(dataset.features) > 5:
            print(f"  ... and {len(dataset.features) - 5} more features")

    return stats

def visualize_feature_stats(pb_file_path):
    """Create visualizations from FeatureStats.pb"""
    stats = tfdv.load_stats_binary(pb_file_path)

    # Use TFDV's built-in visualization
    print("Generating TFDV visualization...")
    display_util.display_statistics(stats)

    return stats

def read_schema_diff(pb_file_path):
    """Read SchemaDiff.pb (anomalies) file"""
    print(f"Reading SchemaDiff from: {pb_file_path}")

    with open(pb_file_path, 'rb') as f:
        anomalies = anomalies_pb2.Anomalies()
        anomalies.ParseFromString(f.read())

    print(f"Number of anomalies: {len(anomalies.anomaly_info)}")

    for feature_name, anomaly_info in anomalies.anomaly_info.items():
        print(f"\nFeature: {feature_name}")
        print(f"  Severity: {anomaly_info.severity}")
        print(f"  Description: {anomaly_info.description}")
        print(f"  Reason: {anomaly_info.reason.type}")

    return anomalies

def extract_stats_to_dataframe(pb_file_path):
    """Convert FeatureStats to pandas DataFrame for custom analysis"""
    stats = tfdv.load_stats_binary(pb_file_path)

    rows = []
    for dataset in stats.datasets:
        for feature in dataset.features:
            row = {
                'dataset_name': dataset.name,
                'feature_name': feature.name,
                'feature_type': None,
                'num_examples': dataset.num_examples,
                'missing_count': None,
                'unique_count': None,
                'min_val': None,
                'max_val': None,
                'mean_val': None,
                'std_val': None
            }

            if feature.HasField('num_stats'):
                row['feature_type'] = 'numeric'
                row['missing_count'] = feature.num_stats.num_zeros
                row['min_val'] = feature.num_stats.min
                row['max_val'] = feature.num_stats.max
                row['mean_val'] = feature.num_stats.mean
                row['std_val'] = feature.num_stats.std_dev

            elif feature.HasField('string_stats'):
                row['feature_type'] = 'string'
                row['unique_count'] = feature.string_stats.unique

            rows.append(row)

    df = pd.DataFrame(rows)
    return df

def compare_train_eval_stats(train_pb_path, eval_pb_path):
    """Compare statistics between train and eval splits"""
    print("Comparing train vs eval statistics...")

    train_stats = tfdv.load_stats_binary(train_pb_path)
    eval_stats = tfdv.load_stats_binary(eval_pb_path)

    # Use TFDV's comparison visualization
    display_util.display_statistics(lhs_statistics=eval_stats,
                                  rhs_statistics=train_stats,
                                  lhs_name='EVAL',
                                  rhs_name='TRAIN')
    
# Example usage functions
def main():
    """Main function with examples"""

    # File paths (update these to your actual paths)
    train_stats_path = "/users/amydp/tfx/pipelines/chicago_taxi_native_keras/StatisticsGen/statistics/3/Split-train/FeatureStats.pb"
    eval_stats_path = "/users/amydp/tfx/pipelines/chicago_taxi_native_keras/StatisticsGen/statistics/3/Split-eval/FeatureStats.pb"
    anomalies_path = "/users/amydp/tfx/pipelines/chicago_taxi_native_keras/ExampleValidator/anomalies/5/Split-train/SchemaDiff.pb"

    print("=== TFX Artifact Analysis ===\n")

    # 1. Read and display feature statistics
    print("1. Reading Feature Statistics:")
    print("-" * 40)
    stats = read_feature_stats(train_stats_path)

    print("\n" + "="*50 + "\n")

    # 2. Read anomalies
    print("2. Reading Anomalies/Schema Differences:")
    print("-" * 40)
    try:
        anomalies = read_schema_diff(anomalies_path)
    except Exception as e:
        print(f"Could not read anomalies: {e}")

    print("\n" + "="*50 + "\n")

    # 3. Convert to DataFrame for analysis
    print("3. Converting to DataFrame:")
    print("-" * 40)
    df = extract_stats_to_dataframe(train_stats_path)
    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")

    print("\n" + "="*50 + "\n")

    # 4. Visualizations (uncomment if running in Jupyter)
    print("4. For visualizations, run in Jupyter notebook:")
    print("   visualize_feature_stats(train_stats_path)")
    print("   compare_train_eval_stats(train_stats_path, eval_stats_path)")
    
if __name__ == "__main__":
    main()