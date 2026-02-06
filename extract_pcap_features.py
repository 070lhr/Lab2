#!/usr/bin/env python3
import pandas as pd
import numpy as np
import dpkt
import socket
from scipy.stats import entropy
from collections import Counter
import os
import glob
import sys
from sklearn.utils import shuffle
import concurrent.futures

# ================= 配置区域 =================
INPUT_DIR = '/data/exp/hrliu/CIC2023/pcap/'
OUTPUT_CSV = './ciciot_ddos_50_50_mixed.csv' # 改个名区分一下
LABEL = 1 
TARGET_RATE_MIN = 2000
TARGET_RATE_MAX = 5000
# ===========================================

def calculate_entropy(ip_list):
    if not ip_list: return 0
    counts = np.array(list(Counter(ip_list).values()))
    probs = counts / len(ip_list)
    return entropy(probs, base=2)

def apply_stealthy_mix(df_original):
    """ 
    【核心修改】
    输入：原始 DataFrame (例如 1000 行)
    操作：随机选 50% 保持原样，另外 50% 进行削弱
    输出：混合后的 DataFrame (依然是 1000 行)
    """
    if df_original is None or df_original.empty:
        return pd.DataFrame()
        
    current_mean_rate = df_original['Rate'].mean()
    
    # 如果原始速率本身就很低，就不折腾了，直接返回原样
    if current_mean_rate < TARGET_RATE_MAX: 
        return df_original
    
    # 1. 随机打乱数据，以便随机抽取 50%
    # frac=1 表示抽取 100% (即全量打乱)，reset_index 重置索引
    df_shuffled = df_original.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. 计算切分点 (50% 位置)
    split_point = len(df_shuffled) // 2
    
    # 3. 切分数据
    df_keep_high = df_shuffled.iloc[:split_point].copy() # 前一半：保留高强度
    df_to_modify = df_shuffled.iloc[split_point:].copy() # 后一半：准备削弱
    
    # 4. 对后一半数据进行削弱 (Make Stealthy)
    center_factor = current_mean_rate / ((TARGET_RATE_MIN + TARGET_RATE_MAX) / 2)
    
    # 生成随机因子
    factors = np.random.uniform(low=max(1.1, center_factor - 1.5), 
                                high=center_factor + 1.5, 
                                size=len(df_to_modify))
    
    df_to_modify['Rate'] = df_to_modify['Rate'] / factors
    df_to_modify['Accel'] = df_to_modify['Accel'] / factors # 加速度也要同步变小
    
    # 5. 合并回去
    df_mixed = pd.concat([df_keep_high, df_to_modify], ignore_index=True)
    
    return df_mixed

def process_single_pcap(pcap_path):
    """ 
    工作进程
    """
    filename = os.path.basename(pcap_path)
    pid = os.getpid()
    print(f"[PID {pid}] 开始处理: {filename} ...")
    
    features_list = []
    current_second = -1
    temp_sizes = []
    temp_ips = []
    
    try:
        with open(pcap_path, 'rb') as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except:
                pcap = dpkt.pcapng.Reader(f)
                
            for ts, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except: continue

                if not isinstance(eth.data, dpkt.ip.IP): continue
                ip = eth.data
                
                timestamp = int(ts)
                size = len(buf)
                try:
                    src_ip = socket.inet_ntoa(ip.src)
                except: continue

                if timestamp != current_second:
                    if current_second != -1:
                        rate = len(temp_sizes)
                        if rate > 0:
                            features_list.append({
                                'timestamp': current_second,
                                'Rate': rate,
                                'Size_Std': np.std(temp_sizes),
                                'Entropy': calculate_entropy(temp_ips),
                                'Label': LABEL,
                                'Source_File': filename
                            })
                    current_second = timestamp
                    temp_sizes = []
                    temp_ips = []
                
                temp_sizes.append(size)
                temp_ips.append(src_ip)

            if temp_sizes:
                features_list.append({
                    'timestamp': current_second,
                    'Rate': len(temp_sizes),
                    'Size_Std': np.std(temp_sizes),
                    'Entropy': calculate_entropy(temp_ips),
                    'Label': LABEL,
                    'Source_File': filename
                })

    except Exception as e:
        print(f"[!] 读取 {filename} 失败: {e}")
        return []

    if not features_list:
        return []

    # 1. 转 DataFrame 并计算 Accel
    df = pd.DataFrame(features_list)
    df = df.sort_values('timestamp')
    df['Accel'] = df['Rate'].diff().fillna(0)
    
    # 2. 【调用新的混合函数】替换掉原来的生成逻辑
    # 这一步会返回一个同样大小的 DataFrame，但其中一半已经是低速率了
    df_final_mixed = apply_stealthy_mix(df)
    
    print(f"[PID {pid}] 完成 {filename}: 共 {len(df_final_mixed)} 条 (50% 原始 / 50% 隐蔽)")
    
    # 返回列表以便主进程合并
    return [df_final_mixed]

def main():
    pcap_files = glob.glob(os.path.join(INPUT_DIR, "*.pcap"))
    
    if not pcap_files:
        print(f"[!] 未找到 PCAP 文件")
        return

    max_workers = os.cpu_count() or 4
    print(f"[*] 检测到 {max_workers} 个 CPU 核心，开启多进程处理...")
    print(f"[*] 模式: 50% 原始保留 + 50% 替换为隐蔽样本")

    all_result_dfs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_pcap, f): f for f in pcap_files}
        
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                dfs = future.result()
                all_result_dfs.extend(dfs)
            except Exception as exc:
                print(f"[!] 异常: {exc}")

    print("="*60)
    print("[*] 正在合并数据...")
    
    if not all_result_dfs:
        return

    df_final = pd.concat(all_result_dfs, ignore_index=True)
    df_final = shuffle(df_final, random_state=42)
    
    cols = ['timestamp', 'Rate', 'Size_Std', 'Entropy', 'Accel', 'Label', 'Source_File']
    final_cols = [c for c in cols if c in df_final.columns]
    df_final = df_final[final_cols]

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"[*] 任务完成！")
    print(f"[*] 总样本数: {len(df_final)}")
    print(f"[*] (这应该等于所有 PCAP 提取出的总秒数，没有重复数据)")

if __name__ == "__main__":
    main()