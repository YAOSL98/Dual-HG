import pandas as pd
from sklearn.model_selection import KFold
import os
from sklearn.model_selection import StratifiedKFold
# ========== 参数配置 ==========
csv_path = "all.csv"   # 输入 csv 文件路径
txt_path = "all.txt"   # 输出 txt 文件路径
out_dir = "cv_splits"       # 保存 3-fold 结果的文件夹
n_splits = 3                # 3 折交叉验证

os.makedirs(out_dir, exist_ok=True)

# ========== 1. 读取CSV ==========
df = pd.read_csv(csv_path)

# ========== 2. 保存完整 txt (带 label) ==========
with open(txt_path, "w") as f:
    for _, row in df.iterrows():
        f.write(f"{row['patient']},{row['label']}\n")
print(f"完整数据已保存到: {txt_path}")

# ========== 3. Stratified 3-fold 交叉验证 ==========
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["label"])):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # 从 train 中再划分 val（按类别分层抽样，20%）
    skf_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold)  # 取 1/5 作为 val ≈ 20%
    # 只跑一次，得到 val
    for inner_train_idx, val_idx in skf_val.split(train_df, train_df["label"]):
        val_df = train_df.iloc[val_idx]
        train_df = train_df.iloc[inner_train_idx]
        break

    # 保存 (只保留 patient 列)
    train_df["patient"].to_csv(os.path.join(out_dir, f"fold{fold+1}_train.txt"), 
                               index=False, header=False)
    val_df["patient"].to_csv(os.path.join(out_dir, f"fold{fold+1}_val.txt"), 
                             index=False, header=False)
    test_df["patient"].to_csv(os.path.join(out_dir, f"fold{fold+1}_test.txt"), 
                              index=False, header=False)

    # ===== 输出类别分布 =====
    print(f"\n=== Fold {fold+1} 类别分布 ===")
    print("Train:\n", train_df["label"].value_counts().sort_index())
    print("Val:\n", val_df["label"].value_counts().sort_index())
    print("Test:\n", test_df["label"].value_counts().sort_index())

print("\n✅ Stratified 3-fold 数据划分完成！")