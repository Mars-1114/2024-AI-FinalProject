import pandas as pd
import re

# Load the CSV file
file_path = '/Users/wesleyhuang/Desktop/NYCU/AI/cleaned_train.csv'
data = pd.read_csv(file_path)

# 3-a | AVERAGE WORD LENGTH
def avg_word_len(txt):
    '''
    Compute the average length of the words of a given input.
        // argument //
            txt - <str> - the input text
        
        // return //
            eval - <float> - the average length of words
    '''
    words = re.findall(r'\b\w+\b', txt)
    if not words:
        return 0
    total_length = sum(len(word) for word in words)
    avg_length = total_length / len(words)
    return avg_length

# 3-b | AVERAGE SENTENCE LENGTH
def avg_sent_len(txt):
    '''
    Compute the average length of the sentences of a given input.
        // argument //
            txt - <str> - the input text
        
        // return //
            eval - <float> - the average length of sentences
    '''
    sentences = re.split(r'[.!?]', txt)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return 0
    total_words = sum(len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences)
    avg_length = total_words / len(sentences)
    return avg_length

# Apply the functions to the dataframe
data['avg_word_len'] = data['text'].apply(avg_word_len)
data['avg_sent_len'] = data['text'].apply(avg_sent_len)

# Group by 'labels' and calculate the mean for each group
grouped_data = data.groupby('labels').agg({
    'avg_word_len': 'mean',
    'avg_sent_len': 'mean'
}).reset_index()

# Display the grouped data
print(grouped_data)
