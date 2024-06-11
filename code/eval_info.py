import re

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
    long_words = [word for word in words if len(word) > 2]
    total_length = sum(len(word) for word in long_words)
    avg_length = total_length / len(long_words) if len(long_words) != 0 else 0
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
    sentences = re.split(r'[.!?。？！]', txt)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return 0
    total_words = sum(len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences)
    avg_length = total_words / len(sentences)
    return avg_length


