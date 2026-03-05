import os
import re
import natsort
# 文件夹路径
folder_path = '/sharefiles1/yaoshuilian/projects/CVPR2026/hypergraphs_mixed_k9'

# 输出文件路径
output_file = 'output_with_categories.txt'
output_file_2 = 'output.txt'
# 获取文件夹中的所有文件并筛选出 .bin 文件
bin_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.bin')]
bin_files = natsort.natsorted(bin_files)
# 用正则表达式匹配符合标准的文件格式 DC000000x yz
pattern = re.compile(r'^DC000000(\d) (\d+)\.bin$')

# 打开输出文件进行写入
with open(output_file, 'w') as f:
    for filename in bin_files:
        match = pattern.match(filename)
        if match:
            # 提取类别x和yz部分
            category = match.group(1)  # 类别x
            rest = match.group(2)  # yz部分
            # 将格式保存为 DC000000x yz, x
            f.write(f"DC000002{category} {rest},{category}\n")
with open(output_file_2, 'w') as f:
    for filename in bin_files:
        match = pattern.match(filename)
        if match:
            # 提取类别x和yz部分
            category = match.group(1)  # 类别x
            rest = match.group(2)  # yz部分
            # 将格式保存为 DC000000x yz, x
            f.write(f"DC000002{category} {rest}\n")
print(f"文件已保存并格式化，路径为 {output_file}")