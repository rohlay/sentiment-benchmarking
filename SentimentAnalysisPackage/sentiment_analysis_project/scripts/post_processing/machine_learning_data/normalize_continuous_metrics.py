import os
import json
import shutil
import pandas as pd
from shutil import copytree, ignore_patterns
from sentiment_analysis_project.scripts.config_dir import CONFIG_ML, METRICS_ML

def normalize_metric(value, metric_name, max_error):
    if metric_name == 'mean_squared_error':
        return value / max_error**2
    elif metric_name in ['root_mean_squared_error', 'mean_absolute_error']:
        return value / max_error
    elif metric_name in ['r2_score', 'explained_variance_score']:
        return value # (value + 1) / 2
    return value

def normalize_regression_file(file_path, dataset_ranges):
    df = pd.read_csv(file_path, index_col=0)
    metric_name = os.path.splitext(os.path.basename(file_path))[0]

    for dataset_name, row in df.iterrows():
        if dataset_name in dataset_ranges:
            upper_value = dataset_ranges[dataset_name][1] 
            lower_value = dataset_ranges[dataset_name][0] 
            max_error = upper_value - lower_value
            #print(f"Normalizing {dataset_name} with max_error {max_error}")
            df.loc[dataset_name] = row.apply(lambda x: normalize_metric(x, metric_name, max_error))
            #print(f"Normalized values: {df.loc[dataset_name]}")
        else:
            print(f"{dataset_name} does not have a corresponding range in dataset_ranges.")

    df.to_csv(file_path)

def main(identifier, reduced_datasets=False):
    # Load config
    with open(CONFIG_ML, 'r') as f:
        config = json.load(f)

    # Map dataset name to its range
    datasets_key = "datasets_processed_reduced" if reduced_datasets else "datasets_processed"
    # Iterating  list of lists, 
    # dataset[0]: dataset[-1], takes the first element to the last element of each inner list.
    # Example:
    # {'News Titles': [-4, 4], 'News Texts': [-4, 4], 'News TitleAndTexts': [-4, 4], 
    # 'Amazon Reviews': [-1, 1], 'Movie Reviews': [-1, 1], 'N.Y. Editorial': [-1, 1], 'Tweets': [-1, 1]}
    dataset_ranges = {dataset[0]: dataset[-1] for dataset in config[datasets_key]}
    print(dataset_ranges)
    source_path = os.path.join(METRICS_ML, identifier)
    dest_path = os.path.join(METRICS_ML, f"{identifier}_normalized")

    # If the normalized directory already exists, delete it
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)

    # 1. Copy entire folder excluding 'kfold' directories
    copytree(source_path, dest_path, ignore=ignore_patterns('*kfold*'))

    # 2. Normalize regression metrics in copied folder
    for module in ['Machine Learning', 'Deep Learning', 'Transformers', 'All']:
        for stats_type in ['mean', 'median', 'max', 'min']:  # Iterating through each of the stats directories
            stats_path = os.path.join(dest_path, module, 'regression', stats_type)
            if os.path.exists(stats_path):  # Check if the directory exists
                for metric_file in os.listdir(stats_path):
                    if metric_file.endswith('.csv'):
                        normalize_regression_file(os.path.join(stats_path, metric_file), dataset_ranges)


"""
Folders:
* ML_SST5_regression  -> ML_SST5_regression_normalized
* ML_ALL_SST_regression -> ML_ALL_SST_regression_normalized
"""
if __name__ == "__main__":
    # Provide identifier here, for example: 'sample_identifier'
    identifier = 'ML_ALL_SST_regression'
    main(identifier)
