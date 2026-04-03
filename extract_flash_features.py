#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter
import os
import sys
import glob
import concurrent.futures

# ================= 配置区域 =================
# 1. 输入目录 (您指定的 CSV 目录)
INPUT_ROOT_DIR = '/data/exp/hrliu/1998WC/WorldCupCSV/' 

# 2. 输出文件路径 (全量高峰数据集)
OUTPUT_CSV = './flash_event_9dim_full.csv'

# 3. 标签：Flash Event = 0
LABEL = 0

# 4. 速率阈值：只保留每秒请求数 > 500 的样本
RATE_THRESHOLD = 700

# 5. 窗口大小 (用于计算 CV, MA)
WINDOW_SIZE = 5
# ===========================================

def calculate_entropy(ids):
    """ 计算 clientID 的香农熵 """
    if len(ids) == 0: return 0
    counts = np.array(list(Counter(ids).values()))
    probs = counts / len(ids)
    return entropy(probs, base=2)

def process_single_csv(file_path):
    """
    处理单个 CSV 文件的完整流程
    """
    fname = os.path.basename(file_path)
    # pid = os.getpid() # 调试用
    
    features_list = []
    
    try:
        # === 1. 精准读取 CSV ===
        # 根据您的截图，表头是标准的: timestamp, clientID, objectID, size, ...
        # 我们只读需要的 3 列
        try:
            # 这里的 engine='c' 更快，skipinitialspace=True 防止列名有空格
            df = pd.read_csv(file_path, 
                             usecols=['timestamp', 'clientID', 'size'],
                             skipinitialspace=True)
            
            # 清洗列名 (防止 'size ' 这种带空格的情况)
            df.columns = df.columns.str.strip()
            
            # 再次确认列是否存在
            required_cols = {'timestamp', 'clientID', 'size'}
            if not required_cols.issubset(df.columns):
                # 如果找不到列，尝试打印一下看看是什么
                # print(f"[!] {fname} 列名不匹配: {df.columns.tolist()}")
                return []

        except Exception as e:
            print(f"[!] 读取 {fname} 失败 (CSV格式错误?): {e}")
            return []

        # === 2. 按秒聚合 ===
        # 确保时间戳是整数 (防止科学计数法导致的浮点误差)
        df['timestamp'] = df['timestamp'].astype(int)
        
        grouped = df.groupby('timestamp')
        
        for ts, group in grouped:
            rate = len(group)
            
            # 【性能优化】
            # 计算 size 标准差
            size_std = group['size'].std()
            if pd.isna(size_std): size_std = 0
            
            # 计算熵 (这是最耗时的步骤)
            # 为了全量特征，我们必须算
            ip_ent = calculate_entropy(group['clientID'].tolist())
            
            features_list.append({
                'timestamp': ts,
                'Rate': rate,
                'Size_Std': size_std,
                'SIP_Ent': ip_ent,
                'Label': LABEL
            })
            
        if not features_list:
            return []
            
        # === 3. 计算时序特征 (关键步骤) ===
        # 必须先转 DataFrame 并按时间排序
        df_res = pd.DataFrame(features_list)
        df_res = df_res.sort_values('timestamp')
        
        # --- 计算衍生特征 ---
        
        # Rate Accel (加速度)
        df_res['Rate_Accel'] = df_res['Rate'].diff().fillna(0)
        
        # Rate CV (变异系数) - 您的核心特征
        # CV = Std / Mean
        r_std = df_res['Rate'].rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)
        r_mean = df_res['Rate'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(0)
        df_res['Rate_CV'] = r_std / (r_mean + 1e-6)
        
        # Entropy Features (熵变化)
        df_res['SIPEnt_Change'] = df_res['SIP_Ent'].diff().fillna(0)
        df_res['SIPEnt_MA'] = df_res['SIP_Ent'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(df_res['SIP_Ent'])
        
        # Size Features (载荷变化)
        df_res['SizeStd_Change'] = df_res['Size_Std'].diff().fillna(0)
        df_res['SizeStd_MA'] = df_res['Size_Std'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(df_res['Size_Std'])
        
        df_res = df_res.fillna(0)
        
        # === 4. 阈值截断 (只保留高峰) ===
        # 这一步会丢弃低流量数据
        df_filtered = df_res[df_res['Rate'] > RATE_THRESHOLD].copy()
        
        if len(df_filtered) > 0:
            return [df_filtered]
        else:
            return []

    except Exception as e:
        # print(f"[!] 处理 {fname} 逻辑错误: {e}")
        return []

def main():
    # 1. 扫描所有 .csv 文件
    # 注意：recursive=True 会搜索子文件夹
    search_pattern = os.path.join(INPUT_ROOT_DIR, '**', '*.csv')
    print(f"[*] 正在扫描目录: {INPUT_ROOT_DIR}")
    
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"[!] 错误: 未找到任何 .csv 文件。请确认路径是否正确。")
        return

    print(f"[*] 找到 {len(files)} 个 CSV 文件。")
    print(f"[*] 正在挖掘 Rate > {RATE_THRESHOLD} 的高峰 Flash 样本...")
    print("="*60)

    all_dfs = []
    
    # 2. 多进程并行处理
    # 根据您的 CPU 核心数自动调整
    max_workers = os.cpu_count() or 4
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_csv, f): f for f in files}
        
        completed_count = 0
        total_files = len(files)
        
        for future in concurrent.futures.as_completed(future_to_file):
            completed_count += 1
            
            # 简单的进度显示
            if completed_count % 50 == 0 or completed_count == total_files:
                sys.stdout.write(f"\r[*] 进度: {completed_count}/{total_files} 文件已处理...")
                sys.stdout.flush()
                
            try:
                res_dfs = future.result()
                if res_dfs:
                    all_dfs.extend(res_dfs)
            except Exception as exc:
                pass

    print("\n" + "="*60)
    
    if not all_dfs:
        print("[!] 警告: 未提取到任何数据。可能原因：")
        print("    1. CSV 格式不匹配 (列名不对?)")
        print("    2. 所有文件的流量都低于阈值 (Rate < 500)")
        return

    # 3. 合并与保存
    print(f"[*] 正在合并 {len(all_dfs)} 个高峰数据片段...")
    df_final = pd.concat(all_dfs, ignore_index=True)
    
    # 整理列顺序
    cols = [
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA', 
        'SIP_Ent',  'SIPEnt_MA', 'SIPEnt_Change',
        'Rate', 'Rate_Accel', 'Rate_CV',    # 注意这里变成了 Rate_CV
        'Label'
    ]
    # 确保列存在
    final_cols = [c for c in cols if c in df_final.columns]
    df_final = df_final[final_cols]
    
    # 保存
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print(f"[*] 任务完成！")
    print(f"[*] 文件已保存至: {os.path.abspath(OUTPUT_CSV)}")
    print(f"[*] 总样本数: {len(df_final)}")
    print("-" * 30)
    # 打印统计信息，确认数据质量
    print(df_final.describe().loc[['count', 'mean', 'min', 'max']])

if __name__ == "__main__":
    main()