import numpy as np
import math
import csv
import time

IGNORE = "!?.。？！"

# This predictor uses -frequency- as the evaluation
# note: you can add any idea you can think of, and feel free to add any packages you need

def resize(table):
    '''
    ## RESIZED RANK
    The rank is based on Zipf's Law, which claims the frequency of a word is inversely proportional to the rank it is in.
    So, instead of giving an evenly spaced rank (diff = 1), we use harmonic series (diff = 1/n),
    which can be approximated to a natural log (we *can* use the actual harmonic series but this is much simpler to implement)
    
        // argument //
            table - <list> - the frequency table
        
        // return //
            rank - <list> - the rank-adjusted frequency table
                format: [element, rank] - <str, float>
    '''
    global memo
    rank = []
    for n in range(len(table)):
        if n < 50:
            if memo[n] == 0:
                memo[n] = math.log(n + 1) + 1
            rank.append([table[n], memo[n]])
        else:
            rank.append([table[n], 4.9])
    return rank

# 0-b | BUILD LETTER FREQUENCY TABLE
def char_freq_table(file):
    '''
    ## Construct letter frequncy table
    You should sort the letter from the most common to the least common
        // argument //
            file - <str> - name of the csv file
            
        // return //
            table - <dict> - a table that list out the frequency rank of letters for all languages
                format: {language id : list of chars}
                e.g. {'en': ['a', 'b', 'c', ...], ...}
    '''
    table = dict()
    
    # import file
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = data[1:]
    
    # count the number of letters
    for train in data:
        lang = train[0]
        txt = train[1]
        if lang not in table:
            table[lang] = dict()
        for char in txt:
            key = char.lower()
            if key not in IGNORE and key != " ":
                if key not in table[lang]:
                    table[lang][key] = 0
                table[lang][key] += 1
    
    # sort the frequency count
    for lang in table:
        temp = list()
        for char in table[lang]:
            temp.append([char, table[lang][char]])
        temp.sort(key = lambda x: x[1], reverse = True)
        table[lang] = [c[0] for c in temp]
        
    #print(table['es'])
    
    return table

# 0-c | BUILD 2-GRAM FREQUENCY TABLE
# note: 2-gram means a length-2 string (e.g. 'ab', 'qu', 't ')
# One important thing is that the space character (' ') should also count, so that we
# can find the most common ending / starting letter.
def twog_freq_table(file):
    '''
    ## Construct 2-gram frequncy table\n
    You should sort the 2-gram from the most common to the least common
        // argument //
            file - <str> - name of the csv file
            
        // return //
            table - <dict> - a table that list out the frequency rank of 2-grams for all languages
                format: {language id : list of 2-grams}
                e.g. {'en': ['ab', 'gh', 'ck', ...], ...}
    '''
    table = dict()
    
    # import file
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = data[1:]
    
    # count the number of 2-grams
    for train in data:
        lang = train[0]
        txt = train[1]
        gram = "  "
        if lang not in table:
            table[lang] = dict()
        # pad an additional space
        if txt[-1] not in IGNORE or txt[-1] != " ":
            txt += " "
        for char in txt:
            key = char.lower()
            if key in IGNORE:
                gram += " "
            else:
                gram += key
            gram = gram[1:]
            if gram != "  ":
                if gram not in table[lang]:
                    table[lang][gram] = 0
                table[lang][gram] += 1
    
    # sort the frequency count
    for lang in table:
        temp = list()
        for char in table[lang]:
            temp.append([char, table[lang][char]])
        temp.sort(key = lambda x: x[1], reverse = True)
        table[lang] = [c[0] for c in temp]
        
    #print(table['es'][:10])
    
    return table

# 2-a | LETTER FREQUENCY CORRELATION
def char_freq_corr(txt, table):
    '''
    ## Find the correlation of the letter frequency ranks between the reference table and the table constructed from input\n
    You can think of the reference rank as the x-coordinate and the test rank as the y-coordinate
        // arguments //
            txt - <str> - the input text\n
            table - <dict> - a table that list out the frequency rank of letters for all languages
                format: {language id : list of chars}
                e.g. {'en': ['a', 'b', 'c', ...], ...}
        
        // return //
            eval_dict - <dict> - a dict of the correlation of two tables in every language
                format: {language id : correlation}
                    # note that -1 <= correlation <= 1
    '''
    eval_dict = dict()
    freq = dict()
    
    # build sampled frequency table
    for char in txt:
        key = char.lower()
        if key not in IGNORE and key != " ":
            if key not in freq:
                freq[key] = 0
            freq[key] += 1
    
    # sort the frequency count
    temp = list()
    for char in freq:
        temp.append([char, freq[char]])
    temp.sort(key = lambda x: x[1], reverse = True)
    freq = [x[0] for x in temp]
    
    # compute error
    for lang in table:
        reference_rank = resize(table[lang])
        sample_rank = resize(freq)
        
        # only take letters that are in the sampled frequency table
        reference_rank = [x for x in reference_rank if x[0] in freq]
        
        # append letters only appear in the sampled table to the reference table
        # the rank is set to negative (this is arbitrary and just to make it that the error would be large)
        for x in freq:
            if x not in table[lang]:
                reference_rank.append([x, math.log(0.1)])
        
        # rearrange the sample rank so that the letters align in both tables
        temp_rank = list()
        for x in reference_rank:
            index = 0
            while sample_rank[index][0] != x[0]:
                index += 1
            temp_rank.append(sample_rank[index])
        sample_rank = temp_rank
        
        # compute error
        xpos = [x[1] for x in reference_rank]
        ypos = [y[1] for y in sample_rank]
        err = sum([abs(xpos[i] - ypos[i]) for i in range(len(xpos))])
        err = err / len(xpos) if len(xpos) != 0 else 0    # standardize
        '''
        xmean = sum(xpos) / len(xpos) if len(xpos) != 0 else 0
        ymean = sum(ypos) / len(ypos) if len(ypos) != 0 else 0
        xdiff = [x - xmean for x in xpos]
        ydiff = [y - ymean for y in ypos]
        cor = np.sum([xdiff[i] * ydiff[i] for i in range(len(xdiff))]) / np.sqrt(np.sum([x ** 2 for x in xdiff]) * np.sum([y ** 2 for y in ydiff]))
        if np.isnan(cor):
            cor = 0
        '''
        
        # store in evaluation
        eval_dict[lang] = err
        
        #if lang == 'zh':
        #    print(reference_rank)
    return eval_dict

# 2-b | 2-GRAM FREQUENCY CORRELATION
def twog_freq_corr(txt, table):
    '''
    ## Find the correlation of the 2-gram frequency ranks between the reference table and the table constructed from input\n
    You can think of the reference rank as the x-coordinate and the test rank as the y-coordinate
        // arguments //
            txt - <str> - the input text\n
            table - <dict> - a table that list out the frequency rank of 2-grams for all languages
                format: {language id : list of 2-grams}
                e.g. {'en': ['a', 'b', 'c', ...], ...}
        
        // return //
            eval_dict - <dict> - a dict of the correlation of two tables in every language
                format: {language id : correlation}
                    # note that -1 <= correlation <= 1
    '''
    eval_dict = dict()
    freq = dict()
    
    # build sampled frequency table
    if txt[-1] not in IGNORE or txt[-1] != " ":
        txt += " "
    gram = "  "
    for char in txt:
        key = char.lower()
        if key in IGNORE:
            gram += " "
        else:
            gram += key
        gram = gram[1:]
        if gram != "  ":
            if gram not in freq:
                freq[gram] = 0
            freq[gram] += 1
    
    # sort the frequency count
    temp = list()
    for twog in freq:
        temp.append([twog, freq[twog]])
    temp.sort(key = lambda x: x[1], reverse = True)
    freq = [x[0] for x in temp]
    
    # compute correlation
    for lang in table:
        reference_rank = resize(table[lang])
        sample_rank = resize(freq)
        
        # only take 2-grams that are in the sampled frequency table
        reference_rank = [x for x in reference_rank if x[0] in freq]
        
        # append 2-grams only appear in the sampled table to the reference table
        # the rank is set to negative (this is arbitrary and just to make it that the correlation would be close to 0 or negative)
        for x in freq:
            if x not in table[lang]:
                reference_rank.append([x, math.log(0.1)])
        
        # rearrange the sample rank so that the 2-grams align in both tables
        temp_rank = list()
        for x in reference_rank:
            index = 0
            while sample_rank[index][0] != x[0]:
                index += 1
            temp_rank.append(sample_rank[index])
        sample_rank = temp_rank
        
        # compute error
        xpos = [x[1] for x in reference_rank]
        ypos = [y[1] for y in sample_rank]
        err = sum([abs(xpos[i] - ypos[i]) for i in range(len(xpos))])
        err = err / len(xpos) if len(xpos) != 0 else 0   # standardize
        '''
        xmean = sum(xpos) / len(xpos) if len(xpos) != 0 else 0
        ymean = sum(ypos) / len(ypos) if len(ypos) != 0 else 0
        xdiff = [x - xmean for x in xpos]
        ydiff = [y - ymean for y in ypos]
        cor = np.sum([xdiff[i] * ydiff[i] for i in range(len(xdiff))]) / np.sqrt(np.sum([x ** 2 for x in xdiff]) * np.sum([y ** 2 for y in ydiff]))
        if np.isnan(cor):
            cor = 0
        '''
        
        # store in evaluation
        eval_dict[lang] = err
    
    return eval_dict

#text = "Letter frequency is the number of times letters of the alphabet appear on average in written language."
#table = twog_freq_table("cleaned_train.csv")
#print(twog_freq_corr(text, table))