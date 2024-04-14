import os
import csv
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
            text = row[1]
            total_words += len(text.split())
            total_chars += len(text)
            total_rows += 1

    # Calculate average number of words, characters, and word length
    avg_words = total_words / total_rows if total_rows else 0
    avg_chars = total_chars / total_rows if total_rows else 0
    avg_word_length = avg_chars / avg_words if avg_words else 0

    return avg_words, avg_chars, avg_word_length, total_rows, total_words, total_chars


def main():
    # Dictionary to store the results
    stats_dict = {}

    # Get all CSV files in the DATA_PROCESSED directory
    csv_files = [f for f in os.listdir(DATA_PROCESSED) if f.endswith('.csv')]

    # Compute statistics for each dataset
    for csv_file in csv_files:
        dataset_name = csv_file.split('-')[0]
        avg_words, avg_chars, avg_word_length, num_rows, total_words, total_chars = read_and_analyze_csv(os.path.join(DATA_PROCESSED, csv_file))
        stats_dict[dataset_name] = (avg_words, avg_chars, avg_word_length, num_rows, total_words, total_chars)

    # Print results in a tabular format
    print("\n{:<15}| {:<10}| {:<10}| {:<12}| {:>12}| {:<12}| {:<12}".format('Dataset', 'Avg Words', 'Avg Chars', 'Word Length', 'Data Pt.', 'Total Words', 'Total Chars'))
    print("{:<15}| {:<10}| {:<10}| {:<12}| {:>12}| {:<12}| {:<12}".format('Name', '/Data Pt.', '/Data Pt.', 'Avg.', '(no. rows)', '', ''))
    print('-' * 100)  # Adjust the line length accordingly
    for dataset, values in stats_dict.items():
        print(f"{dataset:<15}| {values[0]:>10.2f}| {values[1]:>10.2f}| {values[2]:>12.2f}| {values[3]:>12,}| {values[4]:>12,}| {values[5]:>12,}")


if __name__ == "__main__":
    main()
