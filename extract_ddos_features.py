#!/usr/bin/env python3
import pandas as pd
import numpy as np
import dpkt
import socket
from scipy.stats import entropy
from collections import Counter
import os
import glob
from sklearn.utils import shuffle
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 配置区域 =================
# 输入：指定的包含 PCAP 文件的目录路径 (会递归扫描内部所有 .pcap/.pcapng)
INPUT_DIR = '/data/exp/hrliu/CIC2023/pcap' 

# 输出：最终生成的 CSV 文件
OUTPUT_CSV = './ciciot_ddos_9dim_full.csv'

# 标签：DDoS = 1
LABEL = 1 

# 窗口大小
WINDOW_SIZE = 5

# 进程数：默认使用系统所有可用的 CPU 核心数，也可手动指定（如 NUM_WORKERS = 8）
NUM_WORKERS = os.cpu_count()
# ===========================================

def calculate_entropy(ip_list):
    """ 计算 IP 列表的香农熵 """
    if not ip_list: return 0
    counts = np.array(list(Counter(ip_list).values()))
    probs = counts / len(ip_list)
    return entropy(probs, base=2)

def compute_9_features(df):
    """
    【核心特征工程】
    基于 4 个基础特征，扩展计算出 9 个时序特征
    """
    if df.empty: return df
    
    # 确保按时间排序，否则 rolling 和 diff 计算错误
    df = df.sort_values('timestamp')
    
    # === 1. 速率维度 (Rate Dimension) ===
    df['Rate_Accel'] = df['Rate'].diff().fillna(0)
    
    # 加上 1e-6 防止除以 0
    roll_std = df['Rate'].rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)
    roll_mean = df['Rate'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(0)
    df['Rate_CV'] = roll_std / (roll_mean + 1e-6)
    
    # === 2. 熵维度 (Entropy Dimension) ===
    df['SIPEnt_Change'] = df['SIP_Ent'].diff().fillna(0)
    df['SIPEnt_MA'] = df['SIP_Ent'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(df['SIP_Ent'])
    
    # === 3. 载荷维度 (Payload Dimension) ===
    df['SizeStd_Change'] = df['Size_Std'].diff().fillna(0)
    df['SizeStd_MA'] = df['Size_Std'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(df['Size_Std'])
    
    # 清理因 diff 产生的 NaN (填充为 0)
    df = df.fillna(0)
    
    return df

def process_single_pcap(pcap_path):
    """
    处理单个 PCAP 的完整流程
    """
    filename = os.path.basename(pcap_path)
    print(f"[*] 进程 {os.getpid()} 开始处理: {filename} ...")
    
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
                except:
                    continue

                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
                
                timestamp = int(ts)
                size = len(buf)
                
                try:
                    src_ip = socket.inet_ntoa(ip.src)
                except:
                    continue

                # === 时间窗口聚合逻辑 ===
                if timestamp != current_second:
                    if current_second != -1:
                        if len(temp_sizes) > 0:
                            features_list.append({
                                'timestamp': current_second,
                                'Rate': len(temp_sizes),
                                'Size_Std': np.std(temp_sizes),
                                'SIP_Ent': calculate_entropy(temp_ips),
                                'Label': LABEL,
                            })
                    current_second = timestamp
                    temp_sizes = []
                    temp_ips = []
                
                temp_sizes.append(size)
                temp_ips.append(src_ip)
            
            # 处理最后一秒
            if temp_sizes:
                features_list.append({
                    'timestamp': current_second,
                    'Rate': len(temp_sizes),
                    'Size_Std': np.std(temp_sizes),
                    'SIP_Ent': calculate_entropy(temp_ips),
                    'Label': LABEL
                })

    except Exception as e:
        print(f"[!] 读取错误 {filename}: {e}")
        return None

    if not features_list:
        print(f"[!] {filename} 无有效数据。")
        return None

    # 1. 转 DataFrame
    df = pd.DataFrame(features_list)
    
    # 2. 计算 9 个高级时序特征
    df = compute_9_features(df)
    
    print(f"[*] 进程 {os.getpid()} 完成 {filename} 的特征提取，共产生 {len(df)} 条样本")
    return df

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"[!] 错误: 找不到指定的目录 {INPUT_DIR}")
        return

    print("="*60)
    print(f"[*] 目标目录: {INPUT_DIR}")
    print(f"[*] 提取目标: 9 维全量特征 (含 Rate_CV)")
    print(f"[*] 并行处理: 启动 {NUM_WORKERS} 个工作进程")
    print("="*60)

    # 1. 获取所有 pcap / pcapng 文件
    # recursive=True 允许扫描子目录。如果文件全在同一层，可略微加快速度。
    pcap_files = []
    for ext in ('*.pcap', '*.pcapng'):
        pcap_files.extend(glob.glob(os.path.join(INPUT_DIR, '**', ext), recursive=True))

    if not pcap_files:
        print(f"[!] 在目录 {INPUT_DIR} 中没有找到任何 pcap/pcapng 文件。")
        return

    print(f"[*] 扫描到 {len(pcap_files)} 个 PCAP 文件，准备开始处理...\n")

    # 2. 多核并行处理
    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有任务到进程池
        future_to_pcap = {executor.submit(process_single_pcap, path): path for path in pcap_files}
        
        # 收集执行结果
        for future in as_completed(future_to_pcap):
            pcap_path = future_to_pcap[future]
            try:
                df_result = future.result()
                if df_result is not None and not df_result.empty:
                    results.append(df_result)
            except Exception as exc:
                print(f"[!] 处理 {os.path.basename(pcap_path)} 时发生未捕获异常: {exc}")

    # 3. 合并所有数据
    if not results:
        print("\n[!] 未提取到任何有效数据，程序退出。")
        return

    print("\n[*] 正在合并所有提取的数据...")
    df_final = pd.concat(results, ignore_index=True)

    # 全局打乱 (Shuffle)
    print("[*] 正在执行全局数据乱序 (Shuffle)...")
    df_final = shuffle(df_final, random_state=42)
    
    # 整理列顺序
    feature_cols = [
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA', 
        'SIP_Ent',  'SIPEnt_MA', 'SIPEnt_Change',
        'Rate', 'Rate_Accel', 'Rate_CV',
        'Label'
    ]
    
    # 过滤确认列名
    final_cols = [c for c in feature_cols if c in df_final.columns]
    df_final = df_final[final_cols]

    # 保存文件
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print("="*60)
    print(f"[*] 任务成功！")
    print(f"[*] 最终文件保存至: {os.path.abspath(OUTPUT_CSV)}")
    print(f"[*] 总样本数: {len(df_final)}")
    print(f"[*] 特征维度: {len(final_cols)-1} 维 (+Label)")
    print("\n数据预览:")
    print(df_final.head())

if __name__ == "__main__":
    # 在 Windows 系统上多进程要求执行入口受限，这句代码能确保多进程跨平台安全运行
    main()