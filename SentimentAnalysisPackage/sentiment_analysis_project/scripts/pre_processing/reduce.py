import pandas as pd
import sys
import os
from sentiment_analysis_project.scripts.config import DATA_PROCESSED, DATA_REDUCED


def save_reduced_csv(input_csv_path):

    #input_csv_path = DATA_PROCESSED
    output_path = DATA_REDUCED

    # Read the CSV file using pandas
    print("Reading csv file...")
    df = pd.read_csv(input_csv_path)
    print("Done")

    # Check if there are at least 1000 rows
    if len(df) < 1001:
        print(f"The CSV file has only {len(df)} rows. Exiting without creating reduced CSV.")
        return

    # Keep only the first 1000 rows
    df_reduced = df.head(1001)

    # Construct the output file name
    base_name = os.path.basename(input_csv_path)
    output_name = os.path.splitext(base_name)[0].replace('-processed', '-reduced') + '.csv'

    output_path = os.path.join(DATA_REDUCED, output_name)

    # Save the reduced dataframe to a new CSV
    df_reduced.to_csv(output_path, index=False)
    print(f"Saved reduced CSV to: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python reduce_csv.py <input-CSV>')
        exit()

    input_csv_path = sys.argv[1]
    save_reduced_csv(input_csv_path)
