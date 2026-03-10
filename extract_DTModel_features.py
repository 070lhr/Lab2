#!/usr/bin/env python3
import pandas as pd
import numpy as np
import dpkt
import socket
import os
import glob
import concurrent.futures
from sklearn.utils import shuffle

# ================= 配置区域 =================
# 输入目录配置
DDOS_PCAP_DIR = '/data/exp/hrliu/CIC2023/pcap/' 
FLASH_CSV_DIR = '/data/exp/hrliu/1998WC/WorldCupCSV/'

# 输出文件配置
OUTPUT_DDOS_CSV = './tinubu_3dim_ddos.csv'
OUTPUT_FLASH_CSV = './tinubu_3dim_flash.csv'
OUTPUT_MERGED_CSV = './tinubu_3dim_full_mixed.csv'

# WorldCup CSV 列名配置 (请根据您的实际 CSV 表头修改)
# 假设 WorldCup 数据中有时间戳(单位:秒)和客户端ID(代替IP)
CSV_TIME_COL = 'timestamp' 
CSV_IP_COL = 'clientID'

# 标签配置
LABEL_DDOS = 1
LABEL_FE = 0
# ===========================================

def extract_features_from_window(timestamps, ips, seen_ips):
    """
    核心计算逻辑：提取 Tinubu 等人提出的 3 维特征
    """
    if not timestamps:
        return 0, 0, 0.0

    # 1. 源 IP 数 (Num_SrcIP)
    unique_ips = set(ips)
    num_src_ip = len(unique_ips)

    # 2. 新源 IP 数 (Num_New_SrcIP)
    new_ips = unique_ips - seen_ips
    num_new_src_ip = len(new_ips)
    
    # 更新历史 IP 集合
    seen_ips.update(unique_ips)

    # 3. 数据包到达时间间隔平均值 (Mean_IAT)
    if len(timestamps) > 1:
        # 计算相邻数据包的时间差
        iats = np.diff(timestamps)
        mean_iat = np.mean(iats)
    else:
        # 窗口内只有一个包时，IAT 为 0
        mean_iat = 0.0

    return num_src_ip, num_new_src_ip, mean_iat


def process_ddos_pcap(pcap_path):
    """
    解析 DDoS PCAP 文件并提取 3 维特征
    """
    filename = os.path.basename(pcap_path)
    print(f"[PCAP] 开始处理: {filename} ...")
    
    features_list = []
    seen_ips = set()
    
    current_second = -1
    temp_timestamps = []
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
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    ip = eth.data
                    src_ip = socket.inet_ntoa(ip.src)
                except:
                    continue

                sec = int(ts)
                
                # 时间窗聚合逻辑
                if sec != current_second:
                    if current_second != -1 and temp_timestamps:
                        num_src, num_new, mean_iat = extract_features_from_window(temp_timestamps, temp_ips, seen_ips)
                        features_list.append({
                            'timestamp': current_second,
                            'Num_SrcIP': num_src,
                            'Num_New_SrcIP': num_new,
                            'Mean_IAT': mean_iat,
                            'Label': LABEL_DDOS
                        })
                    current_second = sec
                    temp_timestamps = []
                    temp_ips = []
                
                # 保存高精度时间戳用于计算 IAT，保存 IP 用于计算熵
                temp_timestamps.append(ts)
                temp_ips.append(src_ip)
            
            # 处理文件末尾最后一个窗口
            if temp_timestamps:
                num_src, num_new, mean_iat = extract_features_from_window(temp_timestamps, temp_ips, seen_ips)
                features_list.append({
                    'timestamp': current_second,
                    'Num_SrcIP': num_src,
                    'Num_New_SrcIP': num_new,
                    'Mean_IAT': mean_iat,
                    'Label': LABEL_DDOS
                })

    except Exception as e:
        print(f"[!] 读取错误 {filename}: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(features_list)
    return df


def process_flash_csv(csv_path):
    """
    解析 Flash Event CSV 文件并提取 3 维特征
    """
    filename = os.path.basename(csv_path)
    print(f"[CSV] 开始处理: {filename} ...")
    
    try:a
        df_raw = pd.read_csv(csv_path)
        
        # 验证必需的列是否存在
        if CSV_TIME_COL not in df_raw.columns or CSV_IP_COL not in df_raw.columns:
            print(f"[!] 错误: {filename} 缺少必要的列名 ({CSV_TIME_COL} 或 {CSV_IP_COL})")
            return pd.DataFrame()

        # 确保按时间排序
        df_raw = df_raw.sort_values(CSV_TIME_COL)
        
        # 增加一列表示聚合的“秒”
        df_raw['sec_window'] = df_raw[CSV_TIME_COL].astype(int)
        
        features_list = []
        seen_ips = set()
        
        # 按秒聚合进行处理
        for sec, group in df_raw.groupby('sec_window'):
            timestamps = group[CSV_TIME_COL].values
            ips = group[CSV_IP_COL].values
            
            num_src, num_new, mean_iat = extract_features_from_window(timestamps, ips, seen_ips)
            
            features_list.append({
                'timestamp': sec,
                'Num_SrcIP': num_src,
                'Num_New_SrcIP': num_new,
                'Mean_IAT': mean_iat,
                'Label': LABEL_FE
            })
            
    except Exception as e:
        print(f"[!] 读取错误 {filename}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(features_list)


def main():
    max_workers = os.cpu_count() or 4
    print(f"[*] 开始提取 Tinubu 基准模型的 3 维特征...")
    
    # ================= 1. 处理 DDoS PCAP =================
    pcap_files = glob.glob(os.path.join(DDOS_PCAP_DIR, "*.pcap"))
    ddos_dfs = []
    if pcap_files:
        print(f"\n[*] 发现 {len(pcap_files)} 个 PCAP 文件，开始并行处理...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for df in executor.map(process_ddos_pcap, pcap_files):
                if not df.empty:
                    ddos_dfs.append(df)
        df_ddos_final = pd.concat(ddos_dfs, ignore_index=True)
        df_ddos_final.to_csv(OUTPUT_DDOS_CSV, index=False)
        print(f"[+] DDoS 3维特征已保存至 {OUTPUT_DDOS_CSV} (样本数: {len(df_ddos_final)})")
    else:
        print(f"[!] 未在 {DDOS_PCAP_DIR} 找到 PCAP 文件！")
        df_ddos_final = pd.DataFrame()

    # ================= 2. 处理 Flash Event CSV =================
    csv_files = glob.glob(os.path.join(FLASH_CSV_DIR, "*.csv"))
    flash_dfs = []
    if csv_files:
        print(f"\n[*] 发现 {len(csv_files)} 个 CSV 文件，开始并行处理...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for df in executor.map(process_flash_csv, csv_files):
                if not df.empty:
                    flash_dfs.append(df)
        df_flash_final = pd.concat(flash_dfs, ignore_index=True)
        df_flash_final.to_csv(OUTPUT_FLASH_CSV, index=False)
        print(f"[+] FE 3维特征已保存至 {OUTPUT_FLASH_CSV} (样本数: {len(df_flash_final)})")
    else:
        print(f"[!] 未在 {FLASH_CSV_DIR} 找到 CSV 文件！")
        df_flash_final = pd.DataFrame()

    # ================= 3. 合并打乱 (供模型训练测试) =================
    if not df_ddos_final.empty and not df_flash_final.empty:
        df_merged = pd.concat([df_ddos_final, df_flash_final], ignore_index=True)
        # 全局打乱
        df_merged = shuffle(df_merged, random_state=42)
        
        # 整理列顺序
        final_cols = ['Num_SrcIP', 'Num_New_SrcIP', 'Mean_IAT', 'Label']
        df_merged = df_merged[final_cols]
        
        df_merged.to_csv(OUTPUT_MERGED_CSV, index=False)
        print(f"\n[🚀] 最终混合数据集生成成功！已保存至: {OUTPUT_MERGED_CSV}")
        print(f"[*] 总样本数: {len(df_merged)}")
        print("\n数据预览:")
        print(df_merged.head())

if __name__ == "__main__":
    main()