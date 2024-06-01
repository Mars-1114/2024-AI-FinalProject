import numpy as np
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
    # 讀取 CSV 檔案
    df = pd.read_csv(file)
    # 假設每個語言的字符在單獨的列中，語言ID為列名
    table = {}
    for lang in df.columns:
        # 去除空值並將字符轉為小寫（或根據需求處理）
        chars = [str(char).lower() for char in df[lang].dropna().unique()]
        table[lang] = chars
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
    for lang, chars in table.items():
        lang_count = sum(letter_count.get(char, 0) for char in chars)
        eval_dict[lang] = lang_count / total_letters if total_letters > 0 else 0

    return eval_dict
