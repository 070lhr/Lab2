import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# 输入：之前的 9 维 DDoS 数据 (包含了原始+降速)
INPUT_FILE = './ciciot_ddos_9dim_final.csv'
OUTPUT_FILE = './ciciot_ddos_ultimate_train.csv'

def create_ultimate_dataset():
    print("[*] 正在构建终极对抗训练集...")
    
    # 1. 读取现有数据 (里面已经有 50% 原始 + 50% 降速)
    df_existing = pd.read_csv(INPUT_FILE)
    print(f"    - 现有样本数: {len(df_existing)}")
    
    # 2. 生成"困难模式"样本 (Adversarial Samples)
    # 我们复制一份现有的降速样本，专门用来修改它的包大小特征
    # 逻辑：从现有数据中随机抽 50% 出来进行"变异"
    df_hard = df_existing.sample(frac=0.5, random_state=999).copy()
    
    print("    - 正在生成变异样本 (模拟 Size_Std 欺骗)...")
    
    # === 核心变异逻辑 ===
    # 强行把 DDoS 的 Size_Std 改得跟 Flash Event 一样 (200~1000)
    # 这样模型就不能只看 Size_Std 了，必须被迫去看 Rate_Vol (动力学特征)
    
    # 生成随机欺骗值
    random_std = np.random.uniform(200, 1000, size=len(df_hard))
    
    df_hard['Size_Std'] = random_std
    df_hard['SizeStd_MA'] = random_std # 均值也跟着变
    df_hard['SizeStd_Change'] = np.random.normal(0, 50, size=len(df_hard)) # 假装有波动
    
    # === 关键点：动力学特征 (Rate_Vol, Accel) 保持不变！===
    # 因为这是僵尸网络的"指纹"，攻击者很难改掉机器的死板节奏。
    
    # 3. 合并所有数据
    # 最终数据集构成：
    # - 原始简单样本
    # - 原始降速样本
    # - 新增的困难变异样本 (让模型见过世面)
    df_final = pd.concat([df_existing, df_hard], ignore_index=True)
    df_final = shuffle(df_final, random_state=42)
    
    # 保存
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print("="*50)
    print(f"[*] 终极训练集生成完毕: {OUTPUT_FILE}")
    print(f"[*] 总样本数: {len(df_final)}")
    print(f"[*] 包含: 原始暴力流 + 隐蔽降速流 + 对抗欺骗流 (混合训练)")
    print("="*50)

if __name__ == "__main__":
    create_ultimate_dataset()