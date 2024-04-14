import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentiment_analysis_project.scripts.config_dir import PLOTS_LEX, METRICS_LEX
from sentiment_analysis_project.scripts.post_processing.generate_metric_plots.generate_all_plots import generate_plots
from sentiment_analysis_project.scripts.post_processing.generate_metric_plots.generate_all_subplots import generate_subplots


"""
Folders:
* LEX_ALL
* LEX_SST5
* LEX_all_correctedness

* LEX_ALL_modif
"""
def main(experiment_id='LEX_ALL_modif'):
    # If no specific experiment_id is given, use the latest one
    if experiment_id is None:
        experiment_ids = sorted([folder for folder in os.listdir(METRICS_LEX, folder) if os.path.isdir(os.path.join((METRICS_LEX, folder)))])
        if not experiment_ids:
            print("No experiment results found. Please check {LEXICON_RESULTS_DIR}.")
            return
        experiment_id = experiment_ids[-1]

    # Types of results: original and normalized
    #result_types = ["original"]
    #result_types = ["original", "normalized"]
    result_types = ["normalized"]

    # Metrics to consider
    #original_metrics = ['mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error', 'r2_score', 'explained_variance_score', 'correctedness']
    original_metrics = ['mean_squared_error', 'mean_absolute_error', 'root_mean_squared_error']
    #normalized_metrics = ['normalized_mean_squared_error', 'normalized_mean_absolute_error', 'normalized_root_mean_squared_error', 'r2_score', 'explained_variance_score']
   
    # Loop through result types (original and normalized)
    for result_type in result_types:
        # Use the correct metrics based on result_type
        #metrics = original_metrics if result_type == "original" else normalized_metrics
        metrics = original_metrics
        # A dictionary to store the DataFrames for each metric
        dfs = {}

        # Results directory
        results_dir = os.path.join(METRICS_LEX, experiment_id, result_type)

        # Plots directory
        plots_dir = os.path.join(PLOTS_LEX, experiment_id, result_type)
        os.makedirs(plots_dir, exist_ok=True)

        # Loop through each metric folder
        for metric in metrics:
            # Path to CSV file
            path = os.path.join(results_dir, f"{metric}.csv")
            print(f"\nPath to csv file: {path}")
            # Check if the file exists
            if os.path.exists(path):
                print(f"Path EXISTS")
                # Read the CSV file into a DataFrame
                df = pd.read_csv(path, index_col=0)

                # Add the DataFrame to the dictionary
                dfs[metric] = df

                # Create a subfolder for each metric inside the result type folder
                plot_metric_dir = os.path.join(plots_dir, metric)
                os.makedirs(plot_metric_dir, exist_ok=True)

                # Call the function to create and save the individual plots
                generate_plots(df, plot_metric_dir, metric, result_type)
                print(f"\ndf:")
                print(df)
            else:
                print(f"Path DOESN'T EXIST")

        # After looping through all metrics, if dfs is not empty generate the subplots
        if dfs:
            generate_subplots(dfs, plots_dir, result_type)


if __name__ == '__main__':
    main()
