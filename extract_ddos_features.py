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
import concurrent.futures
from sklearn.utils import shuffle

# ================= 配置区域 =================
# 输入：PCAP 文件所在目录
INPUT_DIR = '/data/exp/hrliu/CIC2023/pcap/' 

# 输出：最终合并的 CSV 文件
OUTPUT_CSV = './ciciot_ddos_9dim_full.csv'
 
# 标签：DDoS = 1
LABEL = 1 

# 窗口大小
WINDOW_SIZE = 5
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
    # f1: Rate (基础值)
    # f2: 速率加速度 (变化快慢)
    df['Rate_Accel'] = df['Rate'].diff().fillna(0)
    
    # f3: 速率变异系数 (Rate_CV) - 替代 Rate_Vol
    # CV = Std / Mean (反映相对波动，消除量级影响)
    roll_std = df['Rate'].rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)
    roll_mean = df['Rate'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(0)
    # 加上 1e-6 防止除以 0
    df['Rate_CV'] = roll_std / (roll_mean + 1e-6)
    
    # === 2. 熵维度 (Entropy Dimension) ===
    # f4: Entropy (基础值)
    # f5: 熵的变化率 (攻击开始/结束时的突变)
    df['SIPEnt_Change'] = df['SIP_Ent'].diff().fillna(0)
    # f6: 熵的移动平均 (5秒均值，消除抖动看长期趋势)
    df['SIPEnt_MA'] = df['SIP_Ent'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(df['SIP_Ent'])
    
    # === 3. 载荷维度 (Payload Dimension) ===
    # f7: Size_Std (基础值)
    # f8: 载荷标准差的变化
    df['SizeStd_Change'] = df['Size_Std'].diff().fillna(0)
    # f9: 载荷标准差的均值 (5秒均值)
    df['SizeStd_MA'] = df['Size_Std'].rolling(window=WINDOW_SIZE, min_periods=1).mean().fillna(df['Size_Std'])
    
    # 清理因 diff 产生的 NaN (填充为 0)
    df = df.fillna(0)
    
    return df

def process_single_pcap(pcap_path):
    """
    工作进程：处理单个 PCAP 的完整流程
    1. 解析 PCAP -> 2. 按秒聚合 -> 3. 计算9特征 (不进行任何削弱)
    """
    filename = os.path.basename(pcap_path)
    pid = os.getpid()
    print(f"[PID {pid}] 开始处理: {filename} ...")
    
    features_list = []
    
    # 临时变量
    current_second = -1
    temp_sizes = []
    temp_ips = []
    
    try:
        # 使用 dpkt 读取
        with open(pcap_path, 'rb') as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except:
                pcap = dpkt.pcapng.Reader(f)
                
            for ts, buf in pcap:
                # 解析以太网帧
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except:
                    continue

                # 确保是 IP 包
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
                
                # 提取基础信息
                timestamp = int(ts)
                size = len(buf)
                
                try:
                    src_ip = socket.inet_ntoa(ip.src)
                except:
                    continue

                # === 时间窗口聚合逻辑 ===
                if timestamp != current_second:
                    if current_second != -1:
                        # 结算上一秒
                        if len(temp_sizes) > 0:
                            features_list.append({
                                'timestamp': current_second,
                                'Rate': len(temp_sizes),           # f1 基础
                                'Size_Std': np.std(temp_sizes),    # f7 基础
                                'SIP_Ent': calculate_entropy(temp_ips), # f4 基础
                                'Label': LABEL,
                            })
                    # 重置下一秒
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
        print(f"[PID {pid}] [!] 读取错误 {filename}: {e}")
        return []

    if not features_list:
        print(f"[PID {pid}] {filename} 无有效数据。")
        return []

    # 1. 转 DataFrame
    df = pd.DataFrame(features_list)
    
    # 2. 计算 9 个高级时序特征 (含 Rate_CV)
    df = compute_9_features(df)
    
    # 3. 不再调用 apply_stealthy_mix，直接返回原始数据
    
    print(f"[PID {pid}] 完成 {filename}: 共 {len(df)} 条样本")
    
    return [df]

def main():
    # 1. 扫描文件
    pcap_files = glob.glob(os.path.join(INPUT_DIR, "*.pcap"))
    
    if not pcap_files:
        print(f"[!] 错误: 在 {INPUT_DIR} 下未找到任何 .pcap 文件！")
        return

    # 2. 自动检测 CPU 核数
    max_workers = os.cpu_count() or 4
    print(f"[*] 检测到 {max_workers} 个 CPU 核心，开始并行处理 {len(pcap_files)} 个文件...")
    print(f"[*] 目标: 提取 9 维全量特征 (含 Rate_CV)")
    print("="*60)

    all_dfs = []

    # 3. 多进程并行执行
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_file = {executor.submit(process_single_pcap, f): f for f in pcap_files}
        
        # 获取结果
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                dfs = future.result()
                all_dfs.extend(dfs)
            except Exception as exc:
                print(f"[!] 处理 {filename} 时发生未捕获异常: {exc}")

    print("="*60)
    print("[*] 所有进程结束，正在合并数据...")
    
    if not all_dfs:
        print("[!] 未产生任何数据。")
        return

    # 4. 合并所有 DataFrame
    df_final = pd.concat(all_dfs, ignore_index=True)
    
    # 5. 全局打乱 (Shuffle)
    df_final = shuffle(df_final, random_state=42)
    
    # 6. 整理列顺序
    feature_cols = [
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA', 
        'SIP_Ent',  'SIPEnt_MA', 'SIPEnt_Change',
        'Rate', 'Rate_Accel', 'Rate_CV',    # 注意这里变成了 Rate_CV
        'Label'
    ]
    
    # 过滤一下 (防止万一某列没算出来报错)
    final_cols = [c for c in feature_cols if c in df_final.columns]
    df_final = df_final[final_cols]

    # 7. 保存
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print(f"[*] 任务成功！")
    print(f"[*] 最终文件保存至: {os.path.abspath(OUTPUT_CSV)}")
    print(f"[*] 总样本数: {len(df_final)}")
    print(f"[*] 特征维度: {len(final_cols)-1} 维 (+Label)")
    print("\n数据预览:")
    print(df_final.head())

if __name__ == "__main__":
    main()