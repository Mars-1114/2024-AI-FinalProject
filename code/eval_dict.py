import pandas as pd

# 定義轉換函數
def transform_text(text):
    text = text.lower()  # 轉換為小寫
    text = ''.join(filter(str.isalpha, text))  # 保留字母字符
    unique_sorted_letters = sorted(set(text))  # 去除重複字母並排序
    return ', '.join(unique_sorted_letters)  # 生成結果字符串

# 讀取原始CSV文件
file_path = '/mnt/data/test.csv'
data = pd.read_csv(file_path)

# 轉換文本並替換原始text欄
data['text'] = data['text'].apply(transform_text)

# 將字母從字符串形式轉換為集合形式，便於合併和去重
data['text'] = data['text'].apply(lambda x: set(x.replace(', ', '')))

# 使用groupby和agg函数來合併相同label的不同行的集合
grouped_data = data.groupby('labels')['text'].agg(lambda x: set.union(*x)).reset_index()

# 將合併後的集合轉換回排序後的字符串
grouped_data['text'] = grouped_data['text'].apply(lambda x: ', '.join(sorted(x)))

# 保存到新的CSV文件，更改檔名為 labeled_test.csv
output_path = '/mnt/data/labeled_test.csv'
grouped_data.to_csv(output_path, index=False)

print("輸出的檔案路徑:", output_path)
