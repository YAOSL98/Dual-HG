# 读取train.txt中的名称列表
with open('cv_splits/fold2_train.txt', 'r') as f:
    train_names = [line.strip() for line in f]  # 直接读取每行并去掉空白字符


# 读取all.txt中的类别信息
category_count = {}

with open('all.txt', 'r') as f:
    for line in f:
        name, category = line.split(',')  # 分割名称和类别
        # name = name.strip()  # 去除名称两端的空白
        category = int(category.strip())  # 获取类别并转换为整数
        
        if name in train_names:  # 如果名称在train.txt中
            if category not in category_count:
                category_count[category] = 0
            category_count[category] += 1

# 输出每个类别的数量
for category, count in category_count.items():
    print(f"类别 {category} 的数量: {count}")

print(122-45)
print(122-29)
