import csv
import sys
import re
from tqdm import tqdm

def check_csv(csv_file_name, preprocessed=False):
    with open(csv_file_name, 'r', encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        lines = list(reader)
        total = len(lines)
        count_text = 0
        count_sentiment = 0
        invalid_line = 0
        count_nan_values = 0
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

        print('\nChecking CSV:', csv_file_name)
        print('Total lines:', total)
        
        for i, line in enumerate(tqdm(lines, desc='Checking', unit='rows')):
            try:
                if preprocessed:
                    sentiment, tweet = line
 
                    # Check for NaN values
                    if pd.isna(sentiment) or pd.isna(tweet):
                        count_nan_values += 1
                        continue

                    if tweet == '':
                        count_text += 1

                    if emoji_pattern.search(tweet) is not None:
                        count_sentiment += 1
                    

                else:
                    pass  # Do nothing, just trying to parse the line
            except Exception as e:
                print(f"Error on line {i+1}: {e}")
                print(f"Content of faulty line: {line}")
                invalid_line += 1

        print(f'\nCheck complete. Found {count_text} empty text, {count_sentiment} records with emojis, {count_nan_values} NaN values, and {invalid_line} invalid lines.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python preprocess_checks.py <raw-CSV>')
        exit()

    csv_file_name = sys.argv[1]
    preprocessed = False  
    check_csv(csv_file_name, preprocessed)
