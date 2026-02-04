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
OUTPUT_CSV = './flash_event_expanded_73_74.csv'

# 目标文件列表：覆盖完整的上升和下降周期
# 注意：文件名必须严格按时间顺序排列
TARGET_FILES = [
    'wc_day73_2.csv', 'wc_day73_3.csv', 'wc_day73_4.csv', # 上升前夕 & 爬坡
    'wc_day73_5.csv', 'wc_day73_6.csv',                   # 爆发 & 峰值
    'wc_day74_1.csv', 'wc_day74_2.csv'                    # 回落 & 尾部
]

# 标签：Flash Event = 0
LABEL = 0

# 阈值筛选 (可选)
# 如果您希望保留完整的起步阶段，可以把这个值设低一点 (比如 0 或 500)
# 设为 0 代表保留所有秒的数据
RATE_THRESHOLD = 0 
# ===========================================

def calculate_entropy(ids):
    """ 计算 clientID 的香农熵 """
    if len(ids) == 0: return 0
    # 为了速度，如果 id 数量巨大，可以考虑抽样，但这里我们全算以保证精度
    counts = np.array(list(Counter(ids).values()))
    probs = counts / len(ids)
    return entropy(probs, base=2)

def extract_expanded_flash(input_dir, file_list, output_path):
    print(f"[*] 准备提取扩展 Flash Event 数据...")
    print(f"[*] 文件范围: {file_list[0]} -> {file_list[-1]}")
    
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
            # 读取 CSV (只读需要的列，节省内存)
            # 适配有表头和无表头的情况
            try:
                df_test = pd.read_csv(fpath, nrows=1)
                if 'timestamp' in df_test.columns:
                    df = pd.read_csv(fpath, usecols=['timestamp', 'clientID', 'size'])
                else:
                    # 假设标准格式: timestamp, clientID, objectID, size...
                    df = pd.read_csv(fpath, usecols=[0, 1, 3], names=['timestamp', 'clientID', 'objectID', 'size', 'method', 'status', 'type', 'server'])
                    df = df[['timestamp', 'clientID', 'size']] # 重命名/重排
            except:
                 df = pd.read_csv(fpath, usecols=[0, 1, 3], names=['timestamp', 'clientID', 'objectID', 'size', 'method', 'status', 'type', 'server'])
                 df = df[['timestamp', 'clientID', 'size']]

            # 按秒聚合
            grouped = df.groupby('timestamp')
            
            for ts, group in grouped:
                rate = len(group)
                
                # 阈值过滤
                if rate > RATE_THRESHOLD:
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

    print(f"\n\n[*] 所有文件处理完毕。开始合并与计算衍生特征...")
    
    if not all_features:
        print("[!] 未提取到任何数据。")
        return

    # 转换为 DataFrame
    df_final = pd.DataFrame(all_features)
    
    # === 关键步骤：按时间排序 ===
    # 因为我们是跨文件读取的，必须确保时间轴是连续的，这样算加速度才准
    df_final = df_final.sort_values('timestamp')
    
    # === 计算加速度 (Accel) ===
    # Accel = 当前速率 - 上一秒速率
    # 这样能捕捉到从 73_2 到 73_6 的爬坡过程
    df_final['Accel'] = df_final['Rate'].diff().fillna(0)
    
    # 整理列顺序
    cols = ['timestamp', 'Rate', 'Size_Std', 'Entropy', 'Accel', 'Label']
    df_final = df_final[cols]
    
    # 保存
    df_final.to_csv(output_path, index=False)
    
    print("="*50)
    print(f"[*] 成功生成扩展数据集: {os.path.abspath(output_path)}")
    print(f"[*] 样本总数: {len(df_final)}")
    print(f"[*] 速率范围: Min={df_final['Rate'].min()}, Max={df_final['Rate'].max()}")
    print("="*50)
    print("预览前5行:")
    print(df_final.head())

if __name__ == "__main__":
    extract_expanded_flash(INPUT_DIR, TARGET_FILES, OUTPUT_CSV)