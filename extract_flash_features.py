#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter
import os
import sys

# ================= 配置区域 =================
INPUT_DIR = '/data/exp/hrliu/1998WC/WorldCupCSV/'
OUTPUT_CSV = './flash_event_9dim_full.csv'

TARGET_FILES = [
    'wc_day73_2.csv', 'wc_day73_3.csv', 'wc_day73_4.csv', 
    'wc_day73_5.csv', 'wc_day73_6.csv',                   
    'wc_day74_1.csv', 'wc_day74_2.csv'                    
]

LABEL = 0
WINDOW_SIZE = 5

# 【新增】速率阈值过滤
# 只有每秒请求数大于此值的样本才会被保留
# 建议设为 500 或 1000，以去除低流量噪声，保留真正的 Flash Crowd 高峰
RATE_THRESHOLD = 300 
# ===========================================

def calculate_entropy(ids):
    if len(ids) == 0: return 0
    counts = np.array(list(Counter(ids).values()))
    probs = counts / len(ids)
    return entropy(probs, base=2)

def compute_9_features(df):
    if df.empty: return df
    df = df.sort_values('timestamp')
    
    # 1. Rate Dimension
    df['Rate_Accel'] = df['Rate'].diff().fillna(0)
    
    roll_std = df['Rate'].rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)
    roll_mean = df['Rate'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(0)
    # Rate_CV (变异系数)
    df['Rate_CV'] = roll_std / (roll_mean + 1e-6)
    
    # 2. Entropy Dimension
    df['Ent_Change'] = df['Entropy'].diff().fillna(0)
    df['Ent_MA'] = df['Entropy'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(df['Entropy'])
    
    # 3. Payload Dimension
    df['SizeStd_Change'] = df['Size_Std'].diff().fillna(0)
    df['SizeStd_MA'] = df['Size_Std'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(df['Size_Std'])
    
    df = df.fillna(0)
    return df

def extract_expanded_flash(input_dir, file_list, output_path):
    print(f"[*] 准备提取扩展 Flash Event 数据 (阈值 > {RATE_THRESHOLD})...")
    
    all_features = []
    total_files = len(file_list)
    
    for i, fname in enumerate(file_list):
        fpath = os.path.join(input_dir, fname)
        sys.stdout.write(f"\r    正在处理文件 [{i+1}/{total_files}]: {fname} ...")
        sys.stdout.flush()
        
        if not os.path.exists(fpath):
            continue
            
        try:
            try:
                df_test = pd.read_csv(fpath, nrows=1)
                if 'timestamp' in df_test.columns:
                    df = pd.read_csv(fpath, usecols=['timestamp', 'clientID', 'size'])
                else:
                    raise ValueError("No header")
            except:
                df = pd.read_csv(fpath, usecols=[0, 1, 3], names=['timestamp', 'clientID', 'objectID', 'size', 'method', 'status', 'type', 'server'])
                df = df[['timestamp', 'clientID', 'size']]

            grouped = df.groupby('timestamp')
            
            for ts, group in grouped:
                rate = len(group)
                
                # 【优化】在这里虽然可以过滤，但为了保证 Accel/Rolling 计算的连续性，
                # 建议先全部提取，最后再过滤。
                # 所以这里暂不 drop，只做记录。
                
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

    print(f"\n\n[*] 基础特征提取完毕。开始计算衍生特征...")
    
    if not all_features:
        print("[!] 未提取到任何数据。")
        return

    df_final = pd.DataFrame(all_features)
    df_final = df_final.sort_values('timestamp')
    
    # 先计算衍生特征 (保证时序连续性)
    df_final = compute_9_features(df_final)
    
    # === 【关键修改】最后执行阈值过滤 ===
    # 这样既保证了 Accel 计算正确，又剔除了低流量噪声
    original_count = len(df_final)
    df_final = df_final[df_final['Rate'] > RATE_THRESHOLD]
    filtered_count = len(df_final)
    
    print(f"[*] 阈值过滤: {original_count} -> {filtered_count} (丢弃了 {original_count - filtered_count} 条 Rate <= {RATE_THRESHOLD} 的样本)")
    
    cols = [
        'Rate', 'Rate_Accel', 'Rate_CV', 
        'Entropy', 'Ent_Change', 'Ent_MA', 
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA', 
        'Label'
    ]
    final_cols = [c for c in cols if c in df_final.columns]
    df_final = df_final[final_cols]
    
    df_final.to_csv(output_path, index=False)
    
    print("="*50)
    print(f"[*] Flash 数据生成完毕: {os.path.abspath(output_path)}")
    print("="*50)
    print(df_final.head())

if __name__ == "__main__":
    extract_expanded_flash(INPUT_DIR, TARGET_FILES, OUTPUT_CSV)