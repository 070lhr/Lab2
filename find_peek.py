#!/usr/bin/env python3
import pandas as pd
import os
import glob
import sys

# ================= 配置 =================
# 您的数据目录
INPUT_DIR = '/data/exp/hrliu/1998WC/WorldCupCSV/'

# 我们主要怀疑的目标 (Day 73 和 Day 74)
TARGET_PATTERN = 'wc_day73_*.csv' 
# 如果怀疑跨天，也可以加上 'wc_day74_*.csv'
# =======================================

def find_peak_in_files():
    # 获取所有 Day 73 的文件
    search_path = os.path.join(INPUT_DIR, TARGET_PATTERN)
    files = sorted(glob.glob(search_path))
    
    # 同时也加上 Day 74 的第一个文件，以防万一
    files += sorted(glob.glob(os.path.join(INPUT_DIR, 'wc_day74_1.csv')))

    if not files:
        print(f"[!] 未找到任何文件: {search_path}")
        return

    print(f"[*] 正在扫描 {len(files)} 个文件，寻找最高峰值...")
    
    global_max_rate = 0
    target_file = ""
    target_timestamp = 0

    for f in files:
        filename = os.path.basename(f)
        print(f"    - 正在扫描: {filename} ...", end='\r')
        
        try:
            # 尝试读取 timestamp
            try:
                df = pd.read_csv(f, usecols=['timestamp'])
            except:
                # 无表头兜底
                df = pd.read_csv(f, usecols=[0], names=['timestamp'], header=None)
            
            # 计算该文件内的最大速率
            # value_counts().max() 就是这一秒内请求数的最大值
            counts = df['timestamp'].value_counts()
            
            if counts.empty: continue
            
            local_max = counts.max()
            local_max_time = counts.idxmax() # 发生峰值的时间戳
            
            # print(f"      [Debug] {filename} 最高速率: {local_max}")

            if local_max > global_max_rate:
                global_max_rate = local_max
                target_file = filename
                target_timestamp = local_max_time
                
        except Exception as e:
            print(f"\n[!] 读取 {filename} 出错: {e}")

    print("\n" + "="*50)
    print(f"🏆 找到最高峰值！")
    print(f"📂 所在文件: {target_file}")
    print(f"📈 峰值速率: {global_max_rate} req/s")
    print(f"⏰ 发生时间: {pd.to_datetime(target_timestamp, unit='s')} (UTC)")
    print("="*50)
    
    print("\n建议：")
    print(f"请在提取 Flash Event 特征时，重点使用 /data/exp/hrliu/1998WC/WorldCupCSV/{target_file}")

if __name__ == "__main__":
    find_peak_in_files()