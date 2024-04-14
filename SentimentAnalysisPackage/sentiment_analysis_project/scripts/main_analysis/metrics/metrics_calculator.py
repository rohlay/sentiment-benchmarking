"""
TODO:
* Save Confusion Matrix 
* Compute Mathews Correlation Coefficient (MCC)
"""

import os
import json
import numpy as np
import pandas as pd
from sentiment_analysis_project.scripts.main_analysis.metrics.get_metrics import (get_binary_metrics, get_multiclass_metrics, 
                          get_continuous_metrics)

from sentiment_analysis_project.scripts.config_dir import CONFIG_ML


class MetricsCalculator:
    def __init__(self, metrics_dir, calculate_mean=True, calculate_median=True, calculate_max=True, calculate_min = True):
        
        # Statistics to calculate_
        self.calculate_mean = calculate_mean
        self.calculate_median = calculate_median
        self.calculate_max = calculate_max
        self.calculate_min = calculate_min
        self.metric_stats = {}

        # Metrics definitions
        self.binary_metrics = get_binary_metrics()
        self.multiclass_metrics = get_multiclass_metrics()
        self.continuous_metrics = get_continuous_metrics()
        
        # New folder for each run
        self.metrics_dir = metrics_dir

        # Flags for merging results
        self.has_classification = False
        self.has_regression = False


    def calculate_metrics(self, y_test, predictions, is_continuous, is_multiclass, module_key):
        fold_results = {}

        print("\nCalculating metrics...")
        if is_continuous:
            print(f"Continuous metrics")
        elif is_multiclass:
            print(f"Multiclass metrics")
        else:
            print(f"Binary metrics")

        metrics_categories = [
            (is_continuous and not is_multiclass, self.continuous_metrics),
            (not is_continuous and is_multiclass, self.multiclass_metrics),
            (not is_continuous and not is_multiclass, self.binary_metrics) 
        ]

        # calculate all metrics for current fold
        print(f"Calculating metrics for current fold...")
        for is_category, category_metrics in metrics_categories:
            if is_category:
                for metric_name, metric_func in category_metrics.items():
                    score = metric_func(y_test, predictions)
                    fold_results [metric_name] = round(score, 3)

                    print(f"{metric_name} = {fold_results [metric_name]}") 
        print()

        return fold_results 

    def update_results(self, module_key, dataset_name, model_name, fold_results):
        for metric_name, metric_value in fold_results.items():
            # Use setdefault to create nested dictionaries if they don't exist
            self.metric_stats.setdefault(module_key, {}).setdefault(metric_name, {}).setdefault(dataset_name, {}).setdefault(model_name, {
                'values': [],
                'mean': None,
                'median': None,
                'max': None,
                'min': None
            })
            # Append the new fold metric value
            metric_values = self.metric_stats[module_key][metric_name][dataset_name][model_name]
            metric_values['values'].append(metric_value)

    def compute_metric_statistics(self, module_key, dataset_name, model_name):
        
        for metric_name in self.metric_stats[module_key]:

            # metric_values = self.metric_stats[module_key][metric_name][dataset_name][model_name]
            # Check if the metric exists for the dataset and model
            metric_values = self.metric_stats[module_key].get(metric_name, {}).get(dataset_name, {}).get(model_name, None)

            if metric_values is None:
                print(f"No metric '{metric_name}' found for dataset '{dataset_name}' and model '{model_name}'. Skipping...")
                continue
            
            # Calculate statistics
            values = metric_values['values']

            if len(values) == 0:
                print(f"No values found for metric: {metric_name}")
                continue

            if self.calculate_mean:
                metric_values['mean'] = np.mean(values)
            if self.calculate_median:
                metric_values['median'] = np.median(values)
            if self.calculate_max:
                metric_values['max'] = np.max(values)
            if self.calculate_min:
                metric_values['min'] = np.min(values)


    def save_results(self, module_key):
        statistic_types = ['mean', 'median', 'max', 'min']

        #print(self.metric_stats)
        problem_type = "classification" # placeholder (referenced before assignment)
        # Save statistic values
        for module_key_1, metric_dataset_results in self.metric_stats.items():
            for metric_name, dataset_results in metric_dataset_results.items():

                # Determine the category of metrics
                if metric_name in self.binary_metrics or metric_name in self.multiclass_metrics:
                    problem_type = "classification"
                    self.has_classification = True
                elif metric_name in self.continuous_metrics:
                    problem_type = "regression"
                    self.has_regression = True 

                for statistic_type in statistic_types:

                    # Define a path to store metric results
                    results_path = f"{self.metrics_dir}/{module_key_1}/{problem_type}/{statistic_type}"
                    os.makedirs(results_path, exist_ok=True)

                    # Prepare a dictionary to hold the data for CSV
                    data_for_csv = {}

                    for dataset_name, model_results in dataset_results.items():
                        for model_name, result in model_results.items():
 
                            # Get the statistic value for each model
                            stat_value = result[statistic_type]
                            if model_name not in data_for_csv:
                                data_for_csv[model_name] = {}
                            data_for_csv[model_name][dataset_name] = round(stat_value,3)

                    # Convert the dictionary to a dataframe
                    df = pd.DataFrame(data_for_csv)
                    # Save the dataframe to a CSV file
                    df.to_csv(f"{results_path}/{metric_name}.csv", index=True, header=True)

        # Save raw fold scores
        # Saves a csv file for an entire kfold, for a dataset-model pair
        # where rows are k, columns are the metrics according to the dataset type

        kfold_results_path = f"{self.metrics_dir}/{module_key}/{problem_type}/kfold" 
        os.makedirs(kfold_results_path, exist_ok=True)

        for module_key_2, module_results in self.metric_stats.items():
            for metric_name, metric_results in module_results.items():
                for dataset_name, dataset_results in metric_results.items():
                    fold_metrics_data = {}
                    # We loop over each model for this dataset.
                    for model_name, model_results in dataset_results.items():
                        # Retrieve 'values' using get() method
                        values = model_results.get('values', [])
                        if values:
                            fold_metrics_data[model_name] = values

                    # Create a dataframe from our data.
                    df = pd.DataFrame(fold_metrics_data)
                    # Add 1 to the index to have it start at 1 instead of 0.
                    df.index += 1
                    df.index.name = 'fold'
                    # Save our dataframe to a CSV file.
                    df.to_csv(f"{kfold_results_path}/{dataset_name}_{metric_name}.csv", index=True, header=True)

    def merge_all_results(self):
        # Create 'All' directory
        all_dir_path = f"{self.metrics_dir}/All"
        os.makedirs(all_dir_path, exist_ok=True)

        # For each problem type and statistic type, merge the results and save into the 'All' directory
        problem_types = ['classification', 'regression'] if self.has_classification and self.has_regression else ['classification'] if self.has_classification else ['regression']
        statistic_types = ['mean', 'median', 'max', 'min']
        

        with open(CONFIG_ML, 'r') as f:
            config = json.load(f)
        module_keys = [module_name for module_name, module_info in config['modules'].items() if module_info['active']]

        # module_keys = ['Machine Learning', 'Deep Learning', 'Transformers']


        for problem_type in problem_types:
            for statistic_type in statistic_types:
                for metric_name in self.metric_stats[module_keys[0]].keys():  # Use the first module key to get the metric names
                    # Create an empty DataFrame for each metric
                    final_df = pd.DataFrame()
                    for module_key in module_keys:
                        metric_data = self.metric_stats[module_key][metric_name]
                        for dataset_name, dataset_data in metric_data.items():
                            for model_name, model_data in dataset_data.items():
                                # Retrieve statistic
                                statistic_value = model_data.get(statistic_type, None)
                                if statistic_value is not None:
                                    # Add statistic to DataFrame
                                    final_df.at[dataset_name, f"{model_name}"] = statistic_value

                    # If the DataFrame is not empty, save it as a CSV file
                    if not final_df.empty:
                        dst_file_path = os.path.join(all_dir_path, f"{problem_type}/{statistic_type}/{metric_name}.csv")
                        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                        final_df.to_csv(dst_file_path)
                    else:
                        print(f"No data to merge for problem type '{problem_type}', statistic type '{statistic_type}', and metric '{metric_name}'")
