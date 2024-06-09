import pandas as pd

# 0-a | BUILD DICTIONARY TABLE
def dict_table(file):
    '''
    Construct letter dictionaries for different languages
        // argument //
            file - <str> - name of the csv file
        
        // return //
            table - <dict> - a table contains every letter appears in different languages
                format: {language id : list of chars}
                e.g. {'en': ['a', 'b', 'c', ...], ...}
    '''
    # 定義轉換函數
    def transform_text(text):
        text = text.lower()  # 轉換為小寫
        text = ''.join(filter(str.isalpha, text))  # 保留字母字符
        unique_sorted_letters = sorted(set(text))  # 去除重複字母並排序
        return ', '.join(unique_sorted_letters)  # 生成結果字符串

    # 讀取原始CSV文件
    data = pd.read_csv(file)

    # 轉換文本並替換原始text欄
    data['text'] = data['text'].apply(transform_text)

    # 將字母從字符串形式轉換為集合形式，便於合併和去重
    data['text'] = data['text'].apply(lambda x: set(x.replace(', ', '')))

    # 使用groupby和agg函数來合併相同label的不同行的集合
    grouped_data = data.groupby('labels')['text'].agg(lambda x: set.union(*x)).reset_index()

    # 將合併後的集合轉換回排序後的字符串
    grouped_data['text'] = grouped_data['text'].apply(lambda x: ', '.join(sorted(x)))

    # convert grouped_data into a dict
    table = grouped_data.set_index('labels').T.to_dict('list')
    for lang in table:
        table[lang] = table[lang][0].split(", ")
    return table

# 1 | LANGUAGE PROPORTION
def lang_percent(txt, table):
    '''
    Find the proportion of letters that is in the character set of a language
        // arguments //
            txt - <str> - the input text
            table - <dict> - a table contains every letter appears in different languages
                format: {language id : list of chars}
                e.g. {'en': ['a', 'b', 'c', ...], ...}
        
        // return //
            eval_dict - <dict> - a dict of the proportion of letters in every language
                format: {language id : percentage}
                    # note that 0 <= percentage <= 1
    '''
    # 將輸入文本轉為小寫並計算每個字母的出現次數
    txt = txt.lower()
    letter_count = {char: txt.count(char) for char in set(txt) if char.isalpha()}
    total_letters = sum(letter_count.values())

    eval_dict = {}
    for lang in table:
        lang_count = sum(letter_count.get(char, 0) for char in table[lang])
        eval_dict[lang] = lang_count / total_letters if total_letters > 0 else 0

    return eval_dict

#table = dict_table('cleaned_train.csv')
#print(lang_percent("Hello everyone my name is Markiplier", table))