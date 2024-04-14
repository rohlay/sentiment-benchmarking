import os
import re
import sys
import spacy
import nltk
import contractions
from nltk.corpus import stopwords
from tqdm import tqdm
import csv
import pandas as pd
from sentiment_analysis_project.scripts.config import DATA_PROCESSED

nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word

def is_valid_word(word):
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet

def preprocess_tweet(tweet):
    processed_tweet = []
    # Remove leading and trailing quotes
    tweet = tweet.strip('"')
    # Convert to lower case
    tweet = tweet.lower()
    # Replace contractions
    tweet = contractions.fix(tweet)
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    # Filter unwanted characters
    #tweet = re.sub(r'[^a-zA-Z0-9\s]', '', tweet)
    tweet = re.sub(r'[^\w\sáéíóúñ]', '', tweet) # not remove spanish alphabet
    words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word) and word not in stop_words:
            token = nlp(word)[0]
            if not token.is_stop:
                processed_tweet.append(token.lemma_)

    return ' '.join(processed_tweet)


def check_empty_strings(processed_file_name):
    # Load processed data into DataFrame
    df = pd.read_csv(processed_file_name)

    # Check if any values in DataFrame are empty strings
    if (df.values == '').any():
        print("There are empty strings in the DataFrame. Deleting respective rows...")

        # Delete rows with empty strings
        df = df.replace('', pd.NA)  # replace empty strings with NA values
        df = df.dropna()  # drop rows with NA values

        # Save DataFrame without rows containing empty strings
        df.to_csv(processed_file_name, index=False)
        print("Rows with empty strings have been deleted.")
    else:
        print("No empty strings found in the DataFrame.")

    return df


def preprocess_csv(csv_file_name, processed_file_name, test_file=False):
    save_to_file = open(processed_file_name, 'w', encoding="utf-8")

    # write the column headers to the file
    save_to_file.write('sentiment,text\n')

    # iso-8859-1 encoding is often used as a catch-all for any undefined encodings
    #  'utf-8-sig' encoding, it should handle the BOM characters and allow the sentiment values 
    # to be correctly processed without raising the "invalid literal for int() with base 10" error.
    # encoding="iso-8859-1
    # encoding="utf-8-sig"
    with open(csv_file_name, 'r', encoding="iso-8859-1") as csv_file:
        reader = csv.reader(csv_file)
        lines = list(reader)
        total = len(lines)
        
        for i, line in enumerate(tqdm(lines, desc='Processing', unit='tweets')):
            try:
                sentiment, tweet = line
                sentiment_ft = float(sentiment)
                processed_tweet = preprocess_tweet(tweet)
                if processed_tweet:  # only write rows with valid processed tweets
                    if not test_file:
                        save_to_file.write('%.4f,%s\n' % (sentiment_ft, processed_tweet)) 
                    else:
                        save_to_file.write('%s\n' % processed_tweet)
            except Exception as e:
                print(f"Error on line {i+1}: {e}")
                print("Offending tweet: ", tweet)

    save_to_file.close()        
    print('\nSaved processed tweets to: %s', processed_file_name)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python preprocess.py <raw-CSV>')
        exit()

    csv_file_name = sys.argv[1]
    print('\nReading...',csv_file_name)

    processed_file_name = sys.argv[1][:-4] + '-processed.csv'
    processed_file_path = os.path.join(DATA_PROCESSED, os.path.basename(processed_file_name))

    print('\nStarting preprocess...')
    preprocess_csv(csv_file_name, processed_file_path, test_file=False)
    print('\nPreprocess complete.')
