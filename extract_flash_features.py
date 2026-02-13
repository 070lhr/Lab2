#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter
import os
import sys

# ================= 配置区域 =================
# 输入目录：WC98 CSV 所在位置
INPUT_DIR = '/data/exp/hrliu/1998WC/WorldCupCSV/'

# 输出文件
OUTPUT_CSV = './flash_event_9dim_full.csv'

# 目标文件列表：覆盖完整的上升和下降周期
# 注意：文件名必须严格按时间顺序排列
TARGET_FILES = [
    'wc_day73_2.csv', 'wc_day73_3.csv', 'wc_day73_4.csv', # 上升前夕 & 爬坡
    'wc_day73_5.csv', 'wc_day73_6.csv',                   # 爆发 & 峰值
    'wc_day74_1.csv', 'wc_day74_2.csv'                    # 回落 & 尾部
]

# 标签：Flash Event = 0
LABEL = 0

# 窗口大小 (用于计算 MA 和 CV)
WINDOW_SIZE = 5
# ===========================================

def calculate_entropy(ids):
    """ 计算 clientID 的香农熵 """
    if len(ids) == 0: return 0
    counts = np.array(list(Counter(ids).values()))
    probs = counts / len(ids)
    return entropy(probs, base=2)

def compute_9_features(df):
    """
    计算完整的 9 维特征
    """
    if df.empty: return df
    
    # 确保按时间排序
    df = df.sort_values('timestamp')
    
    # === 1. 速率维度 (Rate Dimension) ===
    # f1: Rate (基础值)
    # f2: 速率加速度 (Rate_Accel)
    df['Rate_Accel'] = df['Rate'].diff().fillna(0)
    
    # f3: 速率变异系数 (Rate_CV) - 替代原来的 Rate_Vol
    # CV = Std / Mean
    roll_std = df['Rate'].rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)
    roll_mean = df['Rate'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(0)
    # 加 1e-6 防止除以 0
    df['Rate_CV'] = roll_std / (roll_mean + 1e-6)
    
    # === 2. 熵维度 (Entropy Dimension) ===
    # f4: Entropy (基础值)
    # f5: 熵的变化率 (Ent_Change)
    df['Ent_Change'] = df['Entropy'].diff().fillna(0)
    # f6: 熵的移动平均 (Ent_MA)
    df['Ent_MA'] = df['Entropy'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(df['Entropy'])
    
    # === 3. 载荷维度 (Payload Dimension) ===
    # f7: Size_Std (基础值)
    # f8: 载荷标准差的变化 (SizeStd_Change)
    df['SizeStd_Change'] = df['Size_Std'].diff().fillna(0)
    # f9: 载荷标准差的均值 (SizeStd_MA)
    df['SizeStd_MA'] = df['Size_Std'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(df['Size_Std'])
    
    # 清理 NaN
    df = df.fillna(0)
    
    return df

def extract_expanded_flash(input_dir, file_list, output_path):
    print(f"[*] 准备提取扩展 Flash Event 数据 (9维特征)...")
    
    all_features = []
    total_files = len(file_list)
    
    for i, fname in enumerate(file_list):
        fpath = os.path.join(input_dir, fname)
        sys.stdout.write(f"\r    正在处理文件 [{i+1}/{total_files}]: {fname} ...")
        sys.stdout.flush()
        
        if not os.path.exists(fpath):
            print(f"\n[!] 警告: 文件 {fname} 不存在，跳过。")
            continue
            
        try:
            # 读取 CSV
            try:
                # 尝试读取有表头的
                df_test = pd.read_csv(fpath, nrows=1)
                if 'timestamp' in df_test.columns:
                    df = pd.read_csv(fpath, usecols=['timestamp', 'clientID', 'size'])
                else:
                    raise ValueError("No header")
            except:
                # 无表头，按标准列索引读取
                df = pd.read_csv(fpath, usecols=[0, 1, 3], names=['timestamp', 'clientID', 'objectID', 'size', 'method', 'status', 'type', 'server'])
                df = df[['timestamp', 'clientID', 'size']]

            # 按秒聚合
            grouped = df.groupby('timestamp')
            
            for ts, group in grouped:
                rate = len(group)
                
                # 计算基础特征
                size_std = group['size'].std()
                if pd.isna(size_std): size_std = 0
                
                ip_ent = calculate_entropy(group['clientID'].tolist())
                
                all_features.append({
                    'timestamp': ts,
                    'Rate': rate,
                    'Size_Std': size_std,
                    'Entropy': ip_ent,
                    'Label': LABEL
                })
                    
        except Exception as e:
            print(f"\n[!] 处理 {fname} 时出错: {e}")

    print(f"\n\n[*] 基础特征提取完毕。开始计算衍生特征 (CV, MA, Accel)...")
    
    if not all_features:
        print("[!] 未提取到任何数据。")
        return

    # 1. 转换为 DataFrame
    df_final = pd.DataFrame(all_features)
    
    # 2. 关键步骤：按时间排序
    df_final = df_final.sort_values('timestamp')
    
    # 3. 计算 9 维衍生特征 (含 Rate_CV)
    df_final = compute_9_features(df_final)
    
    # 4. 整理列顺序
    cols = [
        'Rate', 'Rate_Accel', 'Rate_CV', 
        'Entropy', 'Ent_Change', 'Ent_MA', 
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA', 
        'Label'
    ]
    # 确保列存在
    final_cols = [c for c in cols if c in df_final.columns]
    df_final = df_final[final_cols]
    
    # 保存
    df_final.to_csv(output_path, index=False)
    
    print("="*50)
    print(f"[*] Flash 数据生成完毕: {os.path.abspath(output_path)}")
    print(f"[*] 样本总数: {len(df_final)}")
    print(f"[*] 特征列表: {final_cols}")
    print("="*50)
    print(df_final.head())

if __name__ == "__main__":
    extract_expanded_flash(INPUT_DIR, TARGET_FILES, OUTPUT_CSV)