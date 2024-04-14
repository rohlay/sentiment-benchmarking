# Generates in plots/dataset_type/metric/(10 plots for 5 metrics)
# 5 types of plot for each metric, (mean/median/max/min) times
# Generates in plots/dataset_type/subplots/(2 plots)
# and generates subplots (2 plots), (5 subplots in each)
# import necessary modules from config.py
import os
import pandas as pd
from sentiment_analysis_project.scripts.config_dir import PLOTS_ML, METRICS_ML
from sentiment_analysis_project.scripts.post_processing.generate_metric_plots.generate_all_plots import generate_plots
from sentiment_analysis_project.scripts.post_processing.generate_metric_plots.generate_all_subplots import generate_subplots

"""
Folders:
* ML_SST5_regression
* ML_ALL_SST_classification
* Classification_ALL
"""

def main(experiment_id = 'Classification_ALL'):
    # If no specific experiment_id is given, use the latest one
    if experiment_id is None:
        experiment_ids = sorted([folder for folder in os.listdir(ML_METRICS_DIR) if os.path.isdir(os.path.join(ML_METRICS_DIR, folder))])
        if not experiment_ids:
            print("No experiment results found. Please check the ML_METRICS_DIR.")
            return
        experiment_id = experiment_ids[-1]  # use the latest (last) one

    # Module keys
    module_keys = ["Machine Learning", "Deep Learning", "Transformers", "All"]

    # Set the metric folders
    folders = ["classification","regression"]
    types = ["mean", "median", "max", "min"]

    # Loop through each module_key
    for module_key in module_keys:
        print("Processing module key:", module_key)
        # Experiment results directory
        exp_results_dir = os.path.join(METRICS_ML, experiment_id, module_key)

        # Experiment plots directory
        exp_plots_dir = os.path.join(PLOTS_ML, experiment_id, module_key)
        os.makedirs(exp_plots_dir, exist_ok=True)

        # Loop through each metric folder
        for folder in folders:
            for type in types:
                path = os.path.join(exp_results_dir, folder, type)

                # Create a subfolder for each dataset type inside the plots folder
                plot_folder_dir = os.path.join(exp_plots_dir, folder)
                os.makedirs(plot_folder_dir, exist_ok=True)

                # Check if the folder exists
                if os.path.exists(path):
                    # Get a list of all CSV files in the folder
                    files = [f for f in os.listdir(path) if f.endswith('.csv')]

                    # A dictionary to store the DataFrames for each metric
                    dfs = {}

                    # Loop through each CSV file
                    for file in files:
                        # Read the CSV file into a DataFrame
                        df = pd.read_csv(os.path.join(path, file), index_col=0)
                        metric_name = file.replace(".csv", "")
                        
                        # Add the DataFrame to the dictionary
                        dfs[metric_name] = df

                        # Create a subfolder for each metric type inside the dataset type folder
                        plot_metric_dir = os.path.join(plot_folder_dir, metric_name)
                        os.makedirs(plot_metric_dir, exist_ok=True)

                        # Call the functions from the modules to create and save the plots
                        # Call the functions from the modules to create and save the plots
                        generate_plots(df, plot_metric_dir, metric_name, type)
                        print(f"\ndf:")
                        print(df)
                        print("Number of DataFrames in dfs:", len(dfs))
                    
                    # After looping through all metrics, if dfs is not empty generate the subplots
                    if dfs:
                        generate_subplots(dfs, plot_folder_dir, type)

if __name__ == '__main__':
    main()
