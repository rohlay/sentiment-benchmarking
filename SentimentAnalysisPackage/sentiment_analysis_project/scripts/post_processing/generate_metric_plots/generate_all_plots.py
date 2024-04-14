import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def generate_plots(df, output_dir, metric_name, type):

    # FOR ALL MODELS - ALL DATASETS
    # Create a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.astype(float), annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title(f"Heatmap for {metric_name} in {type} data")
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
    plt.savefig(os.path.join(output_dir, f"{metric_name}_heatmap_{type}.png"), dpi=300)

    # Set the style to "whitegrid" before generating violin plots
    #sns.set_style("whitegrid")

    # FOR EACH MODEL
    # Create a violinplot for each model combined with a boxplot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, inner="quart")
    #sns.boxplot(data=df, color='black')
    plt.title(f"Violin plot for {metric_name} in {type} data (Models)")
    plt.grid(True)
    #plt.xticks(rotation=90)  # To avoid overlapping x-axis labels
    plt.xticks(rotation=45)  # Rotates x-axis labels to 45 degrees
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric_name}_violinplot_models_{type}.png"), dpi=300)

    # FOR EACH DATASET
    # Create a violinplot for each dataset combined with a boxplot
    df_t = df.transpose()
    df_t = df_t.astype(float) # Convert entire DataFrame to float
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_t, inner="quart")
    #sns.boxplot(data=df_t, color='black')
    plt.title(f"Violin plot for {metric_name} in {type} data (Datasets)")
    plt.grid(True)
    #plt.xticks(rotation=90)  # To avoid overlapping x-axis labels
    plt.xticks(rotation=45)  # Rotates x-axis labels to 45 degrees
    plt.savefig(os.path.join(output_dir, f"{metric_name}_violinplot_datasets_{type}.png"), dpi=300)

    # Reset the style to "dark"
    #sns.set_style("dark")

    # Close the plots to save memory
    plt.close('all')
