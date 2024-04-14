import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_subplots(dfs, output_dir, type):
    num_metrics = len(dfs.keys())

    fig, axes = plt.subplots(num_metrics, 3, figsize=(15, num_metrics*5))  # Update to create 3 subplots per metric: heatmap, two violin plots

    for i, metric_name in enumerate(dfs.keys()):
        df = dfs[metric_name]

        # Create a heatmap
        sns.heatmap(df.astype(float), annot=True, cmap='YlGnBu', fmt=".2f", ax=axes[i, 0])
        axes[i, 0].set_title(f"Heatmap for {metric_name}")

        # Create a violinplot for each model
        axes[i, 1].xaxis.set_tick_params(rotation=45)
        sns.violinplot(data=df, inner="quart", ax=axes[i, 1]) # , color='lightgray'
        #sns.boxplot(data=df, ax=axes[i, 1]) #, color='black'
        axes[i, 1].set_title(f"Violin plot for {metric_name} (Models)")
        axes[i, 1].grid(True)

        # Create a violinplot for each dataset
        df_t = df.transpose()
        axes[i, 2].xaxis.set_tick_params(rotation=45)
        sns.violinplot(data=df_t, inner="quart", ax=axes[i, 2]) # , color='lightgray'
        #sns.boxplot(data=df_t, ax=axes[i, 2]) #, color='black'
        axes[i, 2].set_title(f"Violin plot for {metric_name} (Datasets)")
        axes[i, 2].grid(True)

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    os.makedirs(os.path.join(output_dir, "subplots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "subplots", f"subplots_{type}.png"))


