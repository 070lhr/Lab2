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
import concurrent.futures # 引入并行处理库

# ================= 配置区域 =================
INPUT_DIR = '/data/exp/hrliu/CIC2023/pcap/'
OUTPUT_CSV = './ciciot_ddos_all_mixed_parallel.csv'
LABEL = 1 
TARGET_RATE_MIN = 2000
TARGET_RATE_MAX = 5000
# ===========================================

def calculate_entropy(ip_list):
    if not ip_list: return 0
    counts = np.array(list(Counter(ip_list).values()))
    probs = counts / len(ip_list)
    return entropy(probs, base=2)

def generate_stealthy_samples(df_original):
    """ 生成隐蔽样本 (削弱版) """
    if df_original is None or df_original.empty:
        return pd.DataFrame()
        
    current_mean_rate = df_original['Rate'].mean()
    if current_mean_rate < TARGET_RATE_MAX: 
        return pd.DataFrame()
    
    df_stealthy = df_original.copy()
    center_factor = current_mean_rate / ((TARGET_RATE_MIN + TARGET_RATE_MAX) / 2)
    
    # 随机因子
    factors = np.random.uniform(low=max(1.1, center_factor - 1.5), 
                                high=center_factor + 1.5, 
                                size=len(df_stealthy))
    
    df_stealthy['Rate'] = df_stealthy['Rate'] / factors
    df_stealthy['Accel'] = df_stealthy['Accel'] / factors
    
    return df_stealthy

def process_single_pcap(pcap_path):
    """ 
    工作进程：处理单个 PCAP 文件的完整流程 
    (提取特征 -> 计算加速度 -> 生成隐蔽样本)
    """
    filename = os.path.basename(pcap_path)
    # 每个进程都有独立的 ID，方便看日志
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
                # dpkt 解析
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

                # 聚合逻辑
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

            # 最后一秒
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
        return [] # 出错返回空列表

    # 如果没提取到数据
    if not features_list:
        print(f"[PID {pid}] {filename} 无有效数据。")
        return []

    # 1. 转 DataFrame 并计算 Accel
    df_high = pd.DataFrame(features_list)
    df_high = df_high.sort_values('timestamp')
    df_high['Accel'] = df_high['Rate'].diff().fillna(0)
    
    # 2. 生成削弱版数据 (Stealthy)
    df_low = generate_stealthy_samples(df_high)
    
    print(f"[PID {pid}] 完成 {filename}: 原始 {len(df_high)} 条 + 隐蔽 {len(df_low)} 条")
    
    # 返回一个包含两个 DataFrame 的列表 (稍后在主进程合并)
    results = [df_high]
    if not df_low.empty:
        results.append(df_low)
        
    return results

def main():
    pcap_files = glob.glob(os.path.join(INPUT_DIR, "*.pcap"))
    
    if not pcap_files:
        print(f"[!] 未找到 PCAP 文件")
        return

    # 获取 CPU 核心数，决定开多少个进程
    # 比如您有 16 核，就开 16 个进程同时跑
    max_workers = os.cpu_count() or 4
    print(f"[*] 检测到 {max_workers} 个 CPU 核心，即将开启多进程并行处理...")
    print(f"[*] 待处理文件数: {len(pcap_files)}")
    print("="*60)

    all_result_dfs = []

    # === 并行核心代码 ===
    # 使用 ProcessPoolExecutor 自动调度
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # map 会自动把 pcap_files 分发给 process_single_pcap 函数
        # results 是一个生成器，按完成顺序返回结果
        future_to_file = {executor.submit(process_single_pcap, f): f for f in pcap_files}
        
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                # 获取该进程的返回结果 (列表 [df_high, df_low])
                dfs = future.result()
                all_result_dfs.extend(dfs)
            except Exception as exc:
                print(f"[!] 处理 {filename} 时发生异常: {exc}")

    print("="*60)
    print("[*] 所有进程执行完毕，正在合并数据...")
    
    if not all_result_dfs:
        print("[!] 未产生任何数据。")
        return

    # 合并
    df_final = pd.concat(all_result_dfs, ignore_index=True)
    
    # 打乱
    df_final = shuffle(df_final, random_state=42)
    
    # 清理列
    cols = ['timestamp', 'Rate', 'Size_Std', 'Entropy', 'Accel', 'Label', 'Source_File']
    final_cols = [c for c in cols if c in df_final.columns]
    df_final = df_final[final_cols]

    # 保存
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"[*] 最终合并完成！")
    print(f"[*] 文件保存至: {os.path.abspath(OUTPUT_CSV)}")
    print(f"[*] 总样本数: {len(df_final)}")
    print(f"[*] 样本构成: 原始强攻击 (Label=1) + 隐蔽削弱攻击 (Label=1)")

if __name__ == "__main__":
    # Windows/Linux 兼容性保护
    main()