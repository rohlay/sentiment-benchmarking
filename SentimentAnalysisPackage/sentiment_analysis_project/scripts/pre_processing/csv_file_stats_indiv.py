import os
import csv
import sys
from sentiment_analysis_project.scripts.config import DATA_PROCESSED

def read_and_analyze_csv(filename):
    # Initialize counters and totals
    total_words = 0
    total_chars = 0
    total_rows = 0

    # Open the csv file
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip the header
        next(csvreader)

        # Iterate through each row in the csv file
        for row in csvreader:
            # Assuming the format is 'sentiment', 'text', we get the 'text' column by row[1]
            text = row[1]
            
            # Update the total number of words
            total_words += len(text.split())
            
            # Update the total number of characters
            total_chars += len(text)

            # Update the total number of rows
            total_rows += 1

    # Calculate average number of words per data point
    avg_words_per_data_point = total_words / total_rows if total_rows else 0

    # Calculate average number of characters per data point
    avg_chars_per_data_point = total_chars / total_rows if total_rows else 0

    # Calculate average word length
    avg_word_length = avg_chars_per_data_point / avg_words_per_data_point if avg_words_per_data_point else 0


    # Print the results
    print(f"Average number of words per data point: {avg_words_per_data_point:.2f}")
    print(f"Average number of characters per data point: {avg_chars_per_data_point:.2f}")
    print(f"Average word length: {avg_word_length:.2f}")

if __name__ == "__main__":
    csv_file_name = sys.argv[1]

    filename_only = os.path.basename(csv_file_name).split('.')[0]  # Gets 'datasetName-processed' from the full path
    dataset_name = filename_only.split('-')[0]  # Extracts 'datasetName' from 'datasetName-processed'
    print(f"Stats for CSV file: {dataset_name}")

    read_and_analyze_csv(csv_file_name)
