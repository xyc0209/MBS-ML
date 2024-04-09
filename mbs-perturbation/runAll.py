import os

# 设置项目根目录
project_root = 'D:/dataSet/mbs-interference'

# 遍历项目目录
for dirpath, dirnames, files in os.walk(project_root):
    for file in files:
        if file.endswith('.py') and not file.startswith('__'):
            os.system(f"python {os.path.join(dirpath, file)}")