#!/usr/bin/env python3
import pandas as pd
import numpy as np
import dpkt
import socket
import os
import glob
import concurrent.futures
import sys
from sklearn.utils import shuffle

# ================= 配置区域 =================
DDOS_PCAP_DIR = '/data/exp/hrliu/CIC2023/pcap/' 
FLASH_CSV_DIR = '/data/exp/hrliu/1998WC/WorldCupCSV/'

OUTPUT_DDOS_CSV = './tinubu_3dim_ddos.csv'
OUTPUT_FLASH_CSV = './tinubu_3dim_flash.csv'
OUTPUT_MERGED_CSV = './tinubu_3dim_full_mixed.csv'

CSV_TIME_COL = 'timestamp' 
CSV_IP_COL = 'clientID'

LABEL_DDOS = 1
LABEL_FE = 0

# 【极其关键的加速配置】
# 每个 PCAP/CSV 文件最多读取的条目数（1000万条通常已经能覆盖几十分钟的流量，足够证明分布特征了）
# 如果您想硬抗跑完整个文件，可以将其设为 float('inf')
MAX_PACKETS = 10000000 
# ===========================================

def extract_features_from_window(timestamps, ips, seen_ips):
    """ 核心计算逻辑：提取 Tinubu 3维特征 """
    if len(timestamps) == 0:
        return 0, 0, 0.0

    unique_ips = set(ips)
    num_src_ip = len(unique_ips)
    num_new_src_ip = len(unique_ips - seen_ips)
    seen_ips.update(unique_ips)

    if len(timestamps) > 1:
        iats = np.diff(timestamps)
        mean_iat = float(np.mean(iats))
    else:
        mean_iat = 0.0

    return num_src_ip, num_new_src_ip, mean_iat

def process_ddos_pcap(pcap_path):
    """ 采用极限二进制切片法加速解析 PCAP """
    filename = os.path.basename(pcap_path)
    
    features_list = []
    seen_ips = set()
    current_second = -1
    temp_timestamps = []
    temp_ips = []
    
    pkt_count = 0
    
    try:
        with open(pcap_path, 'rb') as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except:
                pcap = dpkt.pcapng.Reader(f)
                
            for ts, buf in pcap:
                pkt_count += 1
                if pkt_count > MAX_PACKETS:
                    print(f"[{filename}] 达到 {MAX_PACKETS} 包截断，提前结束。")
                    break
                
                # 每处理 100 万个包打印一次进度，让您知道没死机
                if pkt_count % 1000000 == 0:
                    print(f"  -> [{filename}] 已飞速处理 {pkt_count//10000} 万个包...")

                # 【核心加速点：二进制直接切片读取，彻底抛弃以太网对象解析】
                # 标准以太网头部 14 字节，偏移 12:14 是协议类型
                try:
                    if len(buf) > 34:
                        eth_type = buf[12:14]
                        if eth_type == b'\x08\x00':  # 纯 IPv4
                            src_ip = socket.inet_ntoa(buf[26:30])
                        elif eth_type == b'\x81\x00' and buf[16:18] == b'\x08\x00': # 带 VLAN 标签的 IPv4
                            src_ip = socket.inet_ntoa(buf[30:34])
                        else:
                            continue # 忽略 ARP、IPv6 等无关流量
                    else:
                        continue
                except:
                    continue

                sec = int(ts)
                
                if sec != current_second:
                    if current_second != -1 and len(temp_timestamps) > 0:
                        num_src, num_new, mean_iat = extract_features_from_window(temp_timestamps, temp_ips, seen_ips)
                        features_list.append({
                            'timestamp': current_second, 'Num_SrcIP': num_src,
                            'Num_New_SrcIP': num_new, 'Mean_IAT': mean_iat, 'Label': LABEL_DDOS
                        })
                    current_second = sec
                    temp_timestamps = []
                    temp_ips = []
                
                temp_timestamps.append(ts)
                temp_ips.append(src_ip)
            
            # 处理最后一个窗口
            if len(temp_timestamps) > 0:
                num_src, num_new, mean_iat = extract_features_from_window(temp_timestamps, temp_ips, seen_ips)
                features_list.append({
                    'timestamp': current_second, 'Num_SrcIP': num_src,
                    'Num_New_SrcIP': num_new, 'Mean_IAT': mean_iat, 'Label': LABEL_DDOS
                })

    except Exception as e:
        print(f"[!] 读取错误 {filename}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(features_list)

def process_flash_csv(csv_path):
    """ 解析 Flash Event CSV """
    filename = os.path.basename(csv_path)
    features_list = []
    seen_ips = set()
    
    try:
        # 使用 nrows 加速截断读取
        df_raw = pd.read_csv(csv_path, skipinitialspace=True, nrows=MAX_PACKETS)
        df_raw.columns = df_raw.columns.str.strip()
        
        if CSV_TIME_COL not in df_raw.columns or CSV_IP_COL not in df_raw.columns:
            return pd.DataFrame()

        df_raw = df_raw.sort_values(CSV_TIME_COL)
        df_raw['sec_window'] = df_raw[CSV_TIME_COL].astype(int)
        
        for sec, group in df_raw.groupby('sec_window'):
            timestamps = group[CSV_TIME_COL].values
            ips = group[CSV_IP_COL].values
            num_src, num_new, mean_iat = extract_features_from_window(timestamps, ips, seen_ips)
            
            features_list.append({
                'timestamp': sec, 'Num_SrcIP': num_src,
                'Num_New_SrcIP': num_new, 'Mean_IAT': mean_iat, 'Label': LABEL_FE
            })
            
    except Exception as e:
        return pd.DataFrame()

    return pd.DataFrame(features_list)

def main():
    # 为了避免 I/O 拥堵，并发数可以稍微限制一下
    max_workers = min(os.cpu_count() or 4, 8) 
    print(f"[*] 开始以极限加速模式提取 Tinubu 的 3 维特征...")
    print(f"[*] 开启多进程: {max_workers} 个 | 文件读取截断阀值: {MAX_PACKETS} 条")
    print("="*60)
    
    # === 1. DDoS PCAP ===
    pcap_files = glob.glob(os.path.join(DDOS_PCAP_DIR, "*.pcap"))
    ddos_dfs = []
    if pcap_files:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for df in executor.map(process_ddos_pcap, pcap_files):
                if not df.empty:
                    ddos_dfs.append(df)
        if ddos_dfs:
            df_ddos_final = pd.concat(ddos_dfs, ignore_index=True)
            df_ddos_final.to_csv(OUTPUT_DDOS_CSV, index=False)
            print(f"\n[+] DDoS 3维特征已保存 (样本数: {len(df_ddos_final)})")
    
    # === 2. Flash Event CSV ===
    csv_files = glob.glob(os.path.join(FLASH_CSV_DIR, "**", "*.csv"), recursive=True)
    flash_dfs = []
    if csv_files:
        print("\n[*] 正在并行处理 FE CSV 文件...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, df in enumerate(executor.map(process_flash_csv, csv_files)):
                if not df.empty:
                    flash_dfs.append(df)
                if (i + 1) % 10 == 0:
                    sys.stdout.write(f"\r  -> 已飞速处理 {i + 1}/{len(csv_files)} 个 CSV...")
                    sys.stdout.flush()
                    
        if flash_dfs:
            df_flash_final = pd.concat(flash_dfs, ignore_index=True)
            df_flash_final.to_csv(OUTPUT_FLASH_CSV, index=False)
            print(f"\n[+] FE 3维特征已保存 (样本数: {len(df_flash_final)})")

    # === 3. 合并打乱 ===
    if ddos_dfs and flash_dfs:
        df_merged = pd.concat([df_ddos_final, df_flash_final], ignore_index=True)
        df_merged = shuffle(df_merged, random_state=42)
        df_merged = df_merged[['Num_SrcIP', 'Num_New_SrcIP', 'Mean_IAT', 'Label']]
        df_merged.to_csv(OUTPUT_MERGED_CSV, index=False)
        print(f"\n[🚀] 最终混合数据集秒速生成成功！已保存至: {OUTPUT_MERGED_CSV}")
        print(f"[*] 总样本数(1秒为1个样本): {len(df_merged)}")

if __name__ == "__main__":
    main()