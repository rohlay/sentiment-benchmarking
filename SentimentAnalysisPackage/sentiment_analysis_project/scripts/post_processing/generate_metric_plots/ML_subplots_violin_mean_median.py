import os
import numpy as np
import pandas as pd
from sentiment_analysis_project.scripts.config_dir import PLOTS_ML, METRICS_ML
import seaborn as sns
import matplotlib.pyplot as plt


def generate_subplots(dfs, output_dir, type):
    num_metrics = len(dfs["mean"].keys())
    fig, axes = plt.subplots(num_metrics, 3, figsize=(15, num_metrics*5))

    for i, metric_name in enumerate(dfs["mean"].keys()):
        print(f"\nGenerating subplots for metric: {metric_name}")

        ### HEATMAP
        # Create a heatmap
        df_mean = dfs["mean"][metric_name]
        print(f"Creating heatmap...")
        sns.heatmap(df_mean.astype(float), annot=True, cmap='YlGnBu', fmt=".2f", ax=axes[i, 0])
        axes[i, 0].set_title(f"Heatmap for {metric_name}")
        print("Heatmap created")

     
        ### VIOLIN 1 MODELS
        print(f"\nPREPARING DATA FOR MODEL VIOLIN PLOT (1)...")
        df_mean = dfs["mean"][metric_name]
        df_median = dfs["median"][metric_name]
        print("\nmean DataFrame:")
        print(df_mean)
        print("\nmedian DataFrame:")
        print(df_median)
        # Add a type column to mean and median dataframes
        df_mean['type'] = 'mean'
        df_median['type'] = 'median'
        # Combine mean and median dataframes and reset index
        df_combined = pd.concat([df_mean, df_median]).reset_index()
        df_combined.rename(columns={'index': 'Dataset'}, inplace=True)
        print("\nCombined DataFrame for MODELS:")
        print(df_combined)

        # Melt the dataframe
        df_melted = df_combined.melt(id_vars=['Dataset', 'type'], var_name='Model', value_name='Score')
        df_melted['Score'] = pd.to_numeric(df_melted['Score'], errors='coerce')
        print("\nMelted DataFrame for MODELS:")
        print(df_melted)

        # Create a combined violinplot for models
        print(f"\nCreating violin plot for MODELS...")
        axes[i, 1].xaxis.set_tick_params(rotation=45)
        sns.violinplot(data=df_melted, x='Model', y='Score', hue='type', split=True, inner="quart", ax=axes[i, 1])
        axes[i, 1].set_title(f"Violin plot for {metric_name} - Models")
        axes[i, 1].grid(True)
        print("Violin plots for MODELS created.")


        ### VIOLIN 2 DATASETS
        print(f"\nPREPARING DATA FOR DATASET VIOLIN PLOT (2)...")
        # Initial data
        df_mean = dfs["mean"][metric_name]
        df_median = dfs["median"][metric_name]

        # Add a type row
        df_mean.loc['type'] = 'mean'
        df_median.loc['type'] = 'median'
        # Remove type column
        df_mean = df_mean.drop(columns=['type'])
        df_median = df_median.drop(columns=['type'])

        print(f"\nInitial data and switching 'type' from column to row...")
        print("\nmean DataFrame:")
        print(df_mean)
        print("\nmedian DataFrame:")
        print(df_median)

        # Transposing
        df_mean_t = df_mean.transpose()
        df_median_t = df_median.transpose()

        # Reset the index 
        df_mean_t.reset_index(inplace=True)
        df_median_t.reset_index(inplace=True)


        print(f"\nTransposing and resetting index...")
        print("\nmean DataFrame transposed:")
        print(df_mean_t)
        print("\nmedian DataFrame transposed:")
        print(df_median_t)
        
        # Rename the column names
        df_mean_t.rename(columns={'index': 'Model'}, inplace=True)
        df_median_t.rename(columns={'index': 'Model'}, inplace=True)

        print(f"\nAfter renaming columns...")
        print(f"\nmean DataFrame transposed:")
        print(df_mean_t)
        print(f"\nmedian DataFrame transposed:")
        print(df_median_t)

        # Combine transposed mean and median dataframes 
        df_t_combined = pd.concat([df_mean_t, df_median_t])
        print("\nCombined DataFrame for DATASETS:")
        print(df_t_combined)

        # Melt the dataframe
        df_t_melted = df_t_combined.melt(id_vars=['Model', 'type'], var_name='Dataset', value_name='Score')
        df_t_melted['Score'] = pd.to_numeric(df_t_melted['Score'], errors='coerce')
        print("\nMelted DataFrame for DATASETS:")
        print(df_t_melted)

        # Create a combined violinplot for each dataset
        print(f"\nCreating violin plot for DATASETS...")
        axes[i, 2].xaxis.set_tick_params(rotation=45)
        sns.violinplot(data=df_t_melted, x='Dataset', y='Score', hue='type', split=True, inner="quart", ax=axes[i, 2])
        axes[i, 2].set_title(f"Violin plot for {metric_name} - Datasets")
        axes[i, 2].grid(True)
        #axes[i, 2].grid(True, color='g', linestyle='-', linewidth=0.5)
        print("Violin plots for DATASETS created.")

    # Adjust the layout
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"combined_subplots.png"), dpi=300)

    # Close the plots to save memory
    plt.close('all')

"""
Folders:
* Classification_ALL
* Regression_ML_ALL_norm
"""

def main(experiment_id = 'Classification_ALL'):
    # If no specific experiment_id is given, use the latest one
    if experiment_id is None:
        experiment_ids = sorted([folder for folder in os.listdir(METRICS_ML) if os.path.isdir(os.path.join(METRICS_ML, folder))])
        if not experiment_ids:
            print("No experiment results found. Please check the METRICS_ML.")
            return
        experiment_id = experiment_ids[-1]  # use the latest (last) one

    # Module keys
    module_keys = ["Machine Learning", "Deep Learning", "Transformers", "All"]

    # Set the metric folders
    folders = ["classification","regression"]

    # Loop through each module_key
    for module_key in module_keys:
        print("Processing module key:", module_key)
        # Results directory for each experiment (to read)
        exp_results_dir = os.path.join(METRICS_ML, experiment_id, module_key)


        # Loop through each metric folder
        for folder in folders:
            combined_dfs = {"mean": {}, "median": {}}  # Keep track of mean and median dfs

            for type in ["mean", "median"]:
                path = os.path.join(exp_results_dir, folder, type)

                # Check if the folder exists
                if os.path.exists(path):
                    # Get a list of all CSV files in the folder
                    files = [f for f in os.listdir(path) if f.endswith('.csv')]

                    # Loop through each CSV file
                    for file in files:
                        # Read the CSV file into a DataFrame
                        df = pd.read_csv(os.path.join(path, file), index_col=0)
                        metric_name = file.replace(".csv", "")
                        
                        # Add the DataFrame to the combined_dfs
                        combined_dfs[type][metric_name] = df

                        print(f"\nMetric name: {metric_name}")
                        print(f"df:")
                        print(df)
                        print("Number of DataFrames in combined_dfs:", len(combined_dfs[type]))
                
                    # Sublots directory for each experiment (to write)
                    subplots_dir = os.path.join(PLOTS_ML, experiment_id, module_key, folder, 'subplots')
                    os.makedirs(subplots_dir, exist_ok=True)

                    # After looping through all metrics, if combined_dfs is not empty generate the subplots
                    if combined_dfs["mean"] and combined_dfs["median"]:
                        generate_subplots(combined_dfs, subplots_dir, type)

if __name__ == '__main__':
    main()




