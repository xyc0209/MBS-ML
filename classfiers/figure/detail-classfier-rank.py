
import pandas as pd

# 从Excel读取数据
data = pd.read_excel('C:/Users/xingy/Desktop/detail-rank.xlsx')
# 合并 'classfier' 和 'sampling' 列为新的一列 'key'
data['key'] = data['classfier'].astype(str) + '-' + data['sampling'].astype(str)
# 按照类型（type）进行分组
grouped = data.groupby('type')

# 对AUC列进行降序排列，并计算排名
data['Rank'] = grouped['AUC'].rank(ascending=False, method='average')
grouped = data.groupby('key')
# 计算每个分组中 'Rank' 列的平均值
# 计算每个分组中 'Rank' 列的和
grouped_sum = grouped['Rank'].sum()
grouped_mean = grouped['Rank'].mean()
# 输出每个类型下的AUC排名

output_data = pd.DataFrame({'Key': [], 'Mean-Rank': []})

# 输出每个键（key）对应的排名之和和平均值
for key, rank_sum in grouped_sum.items():
    rank_mean = grouped_mean[key]
    output_data = output_data.append({'Key': key, 'Mean-Rank': rank_mean}, ignore_index=True)

# 将数据框输出到Excel文件
output_data.to_excel('output_excel_file.xlsx', index=False)