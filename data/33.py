import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

# ========== 参数配置 ==========
csv_path = "all.csv"   # 输入 csv 文件路径
txt_path = "all.txt"   # 输出 txt 文件路径
out_dir = "cv_splits_new"  # 保存交叉验证结果的文件夹
n_splits = 3           # 3 折交叉验证（保留原交叉验证逻辑）

os.makedirs(out_dir, exist_ok=True)

# ========== 1. 读取CSV ==========
df = pd.read_csv(csv_path)

# ========== 2. 保存完整 txt (带 label) ==========
with open(txt_path, "w") as f:
    for _, row in df.iterrows():
        f.write(f"{row['patient']},{row['label']}\n")
print(f"完整数据已保存到: {txt_path}")

# ========== 3. 分层划分 训练:验证:测试 = 4:3:3 并执行3折交叉验证 ==========
# 外层：3折交叉验证（保证不同折的分布一致性）
skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_val_idx, test_idx) in enumerate(skf_outer.split(df, df["label"])):
    # 第一步：取当前折的 训练+验证 集、测试集（测试集占比≈30%）
    train_val_df = df.iloc[train_val_idx]  # 待划分的 训练+验证 集（占比≈70%）
    test_df = df.iloc[test_idx]            # 最终测试集（30%）
    
    # 第二步：将 训练+验证 集 按 4:3 分层划分（即从70%中取4/7作为训练集，3/7作为验证集）
    # n_splits=7：将数据分为7份，取1份作为验证集（3/7），剩余6份作为训练集（4/7），保证4:3比例
    skf_inner = StratifiedKFold(n_splits=7, shuffle=True, random_state=fold+42)  # 随机种子错开，避免重复
    # 只执行一次划分，取第一份作为验证集，剩余作为训练集
    for inner_train_idx, val_idx in skf_inner.split(train_val_df, train_val_df["label"]):
        train_df = train_val_df.iloc[inner_train_idx]  # 最终训练集（40%）
        val_df = train_val_df.iloc[val_idx]            # 最终验证集（30%）
        break  # 仅需一次划分，跳出内层循环

    # ========== 4. 保存各集结果（仅保留patient列，无索引无表头） ==========
    train_save_path = os.path.join(out_dir, f"fold{fold+1}_train.txt")
    val_save_path = os.path.join(out_dir, f"fold{fold+1}_val.txt")
    test_save_path = os.path.join(out_dir, f"fold{fold+1}_test.txt")
    
    train_df["patient"].to_csv(train_save_path, index=False, header=False)
    val_df["patient"].to_csv(val_save_path, index=False, header=False)
    test_df["patient"].to_csv(test_save_path, index=False, header=False)

    # ========== 5. 输出当前折的类别分布和数据量占比（验证4:3:3比例） ==========
    total_samples = len(df)
    train_ratio = f"{len(train_df)/total_samples:.1%}"
    val_ratio = f"{len(val_df)/total_samples:.1%}"
    test_ratio = f"{len(test_df)/total_samples:.1%}"
    
    print(f"\n=== Fold {fold+1} 数据划分结果 ===")
    print(f"数据量占比：训练集{train_ratio} | 验证集{val_ratio} | 测试集{test_ratio}")
    print("训练集类别分布:\n", train_df["label"].value_counts().sort_index())
    print("验证集类别分布:\n", val_df["label"].value_counts().sort_index())
    print("测试集类别分布:\n", test_df["label"].value_counts().sort_index())

print(f"\n✅ 3折交叉验证完成！所有划分结果已保存至 {out_dir} 文件夹，训练:验证:测试=4:3:3 分层划分")