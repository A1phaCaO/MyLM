
import pyarrow.parquet as pq
import pandas as pd
import json

path = repr(input('输入文件地址')).strip("'").strip('"')
parquet_file = pq.ParquetFile(path)
data = parquet_file.read().to_pandas()

df = pd.DataFrame(data)
# 筛选出langdetect为zh-cn的列
df = df[df['langdetect'] == 'zh']

# 只保留conversation列
df = df[['conversations']]

# # 将列中的json数据只保留value和from项
# df['conversations'] = df['conversations'].apply(lambda x: [json.loads(i)['value'] for i in x])


# 将数据保存为csv文件
res_path = './processed_parquet.jsonl'
df.to_json(res_path, orient='records', lines=True, force_ascii=False)
print(f'数据已保存到 {res_path}')

