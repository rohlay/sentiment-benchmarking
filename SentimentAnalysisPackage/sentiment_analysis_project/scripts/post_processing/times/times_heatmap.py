import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentiment_analysis_project.scripts.config import TIMES_DIR

# Assuming you've imported or defined the TIMES_DIR somewhere earlier in your script
ID = 'outputs7'
MEAN_DIR = os.path.join(TIMES_DIR, ID, 'mean')

def create_heatmap(filename):
    # Load the CSV file into a pandas DataFrame
    data_path = os.path.join(MEAN_DIR, filename)
    df = pd.read_csv(data_path, index_col=0)  # Assuming the first column is the row index (dataset names)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))  # Adjust size as needed
    sns.heatmap(df, annot=True, cmap='viridis', fmt='.5g')  # 'annot' adds annotations inside cells, 'fmt' adjusts the decimal formatting
    plt.title(filename.replace(".csv", ""))
    plt.show()

# Creating heatmaps for both files
create_heatmap('train_times_mean.csv')
create_heatmap('predict_times_mean.csv')
