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

# ================= 配置区域 =================
INPUT_DIR = '/data/exp/hrliu/CIC2023/pcap/'
OUTPUT_CSV = './ciciot_ddos_all_mixed_fast.csv'
LABEL = 1 
TARGET_RATE_MIN = 2000
TARGET_RATE_MAX = 5000
# ===========================================

def calculate_entropy(ip_list):
    if not ip_list: return 0
    counts = np.array(list(Counter(ip_list).values()))
    probs = counts / len(ip_list)
    return entropy(probs, base=2)

def extract_features_fast(pcap_path):
    filename = os.path.basename(pcap_path)
    print(f"[*] 正在处理文件 (dpkt模式): {filename} ...")
    
    features_list = []
    
    current_second = -1
    temp_sizes = []
    temp_ips = []
    packet_count = 0
    
    try:
        # 使用 dpkt 必须以二进制模式打开
        with open(pcap_path, 'rb') as f:
            # 自动识别 pcap 或 pcapng
            try:
                pcap = dpkt.pcap.Reader(f)
            except:
                pcap = dpkt.pcapng.Reader(f)
                
            for ts, buf in pcap:
                packet_count += 1
                if packet_count % 500000 == 0: # 这里的刷新频率可以调高，因为dpkt很快
                    sys.stdout.write(f"\r    > 已扫描数据包: {packet_count} ...")
                    sys.stdout.flush()

                # === dpkt 解析逻辑 ===
                # 1. 解析以太网层
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except:
                    continue # 某些损坏的包跳过

                # 2. 确保是 IP 包 (IP type is 2048)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                
                ip = eth.data
                
                # 3. 提取基础信息
                # dpkt 的 ts 可能是 float，强转 int
                timestamp = int(ts) 
                size = len(buf) # 总长度
                
                # dpkt 提取出的 src 是二进制 (b'\xc0\xa8...'), 需要转成字符串
                # 虽然算熵其实用二进制也行，但为了统一习惯转一下
                try:
                    src_ip = socket.inet_ntoa(ip.src)
                except:
                    continue

                # === 时间窗口聚合逻辑 (和之前完全一样) ===
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

            # 处理最后一秒
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
        print(f"\n[!] 读取 {filename} 失败: {e}")
        return None

    print(f"\n    > {filename} 提取完毕，共 {len(features_list)} 秒样本。")
    
    if not features_list: return None

    df = pd.DataFrame(features_list)
    df = df.sort_values('timestamp')
    df['Accel'] = df['Rate'].diff().fillna(0)
    return df

def generate_stealthy_samples(df_original):
    # (这部分代码保持不变，直接复用)
    print("    > 正在生成隐蔽样本 (Low-Rate Data Augmentation)...")
    df_stealthy = df_original.copy()
    current_mean_rate = df_original['Rate'].mean()
    if current_mean_rate < TARGET_RATE_MAX: return pd.DataFrame()
    
    center_factor = current_mean_rate / ((TARGET_RATE_MIN + TARGET_RATE_MAX) / 2)
    factors = np.random.uniform(low=max(1.1, center_factor - 1.5), 
                                high=center_factor + 1.5, 
                                size=len(df_stealthy))
    df_stealthy['Rate'] = df_stealthy['Rate'] / factors
    df_stealthy['Accel'] = df_stealthy['Accel'] / factors
    return df_stealthy

def main():
    # 安装依赖: pip install dpkt
    pcap_files = glob.glob(os.path.join(INPUT_DIR, "*.pcap"))
    pcap_files.sort()
    
    if not pcap_files:
        print(f"[!] 未找到 PCAP 文件")
        return

    all_dfs = []
    print(f"[*] 使用 dpkt 高速引擎处理 {len(pcap_files)} 个文件...")
    
    for pcap_path in pcap_files:
        df_high = extract_features_fast(pcap_path)
        if df_high is not None and not df_high.empty:
            df_low = generate_stealthy_samples(df_high)
            all_dfs.append(df_high)
            if not df_low.empty:
                all_dfs.append(df_low)
        print("-" * 60)

    print("[*] 正在合并...")
    if not all_dfs: return

    df_final = pd.concat(all_dfs, ignore_index=True)
    df_final = shuffle(df_final, random_state=42)
    
    cols = ['timestamp', 'Rate', 'Size_Std', 'Entropy', 'Accel', 'Label', 'Source_File']
    final_cols = [c for c in cols if c in df_final.columns]
    df_final = df_final[final_cols]

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"[*] 完成！文件保存至: {OUTPUT_CSV}")
    print(f"[*] 总样本数: {len(df_final)}")

if __name__ == "__main__":
    main()