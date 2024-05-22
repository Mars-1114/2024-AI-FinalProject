import pandas as pd
import re

def clean_data(file_path):
    # 讀取數據
    df = pd.read_csv(file_path)

    # 1. data cleaning (把有遺漏參數的資料補齊或去除)
    df.dropna(inplace=True)

    # 2. 移除標點符號，除了句末的句號
    def remove_unwanted_chars(text):
        # 移除標點符號，保留句末的句號
        text = re.sub(r'(?<!\.$)[^\w\s.!?。？！]', '', text)  # 移除非句末的標點符號
        text = re.sub(r'(?<!\.$)[\d]', '', text)  # 移除數字
        #text = re.sub(r'\.(?!\s*$)', '', text)  # 移除非句末的句號
        return text

    df['text'] = df['text'].apply(remove_unwanted_chars)

    # 3. 移除多餘空格字元 (但字跟字中間的一個空格要保留)
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    # 4. 匯出成csv檔
    cleaned_file_path = 'cleaned_train.csv'
    df.to_csv(cleaned_file_path, index=False)
    print(f'清理後的數據已存儲為: {cleaned_file_path}')

# 使用函數
clean_data('train.csv')