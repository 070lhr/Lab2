#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scapy.all import PcapReader, IP
from scipy.stats import entropy
from collections import Counter
import os
import sys

# ================= 配置区域 =================
# 输入：指定的绝对路径
INPUT_PCAP = '/data/exp/hrliu/CIC2023/DDoS-HTTP_Flood-.pcap' 

# 输出：保存到当前目录 (./)
OUTPUT_CSV = './ciciot_http_flood_features.csv'

# 标签：CICIoT2023 的攻击样本标记为 1
LABEL = 1 

# 限制：为了防止跑太久，设置最大处理秒数 (例如 3000秒)
# 如果想跑全量数据，请将此值改为 None
MAX_SECONDS = None
# ===========================================

def calculate_entropy(ip_list):
    """ 计算 IP 列表的香农熵 (Shannon Entropy) """
    if not ip_list:
        return 0
    # 统计每个 IP 出现的次数
    counts = np.array(list(Counter(ip_list).values()))
    # 计算概率分布
    probs = counts / len(ip_list)
    # 计算熵 (base 2)
    return entropy(probs, base=2)

def extract_features(pcap_path, output_path, label=1, max_seconds=None):
    print(f"[*] 开始处理 PCAP 文件: {pcap_path}")
    print(f"[*] 目标输出: {output_path}")
    
    if not os.path.exists(pcap_path):
        print(f"[!] 错误: 找不到文件 {pcap_path}")
        print("    请检查路径是否正确，或者是否有读取权限。")
        return

    features_list = []
    
    # 临时缓存：存储当前这一秒的数据
    current_second = -1
    temp_sizes = []
    temp_ips = []
    
    # 计数器
    packet_count = 0
    processed_seconds = 0
    
    try:
        # 使用 PcapReader 进行流式读取 (节省内存)
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                packet_count += 1
                if packet_count % 10000 == 0:
                    # \r 用于在 Linux 终端原地刷新进度
                    sys.stdout.write(f"\r    已扫描数据包: {packet_count} | 已提取秒数: {processed_seconds}...")
                    sys.stdout.flush()
                
                # 我们只关心 IP 层的数据
                if IP in pkt:
                    # 获取时间戳 (取整到秒)
                    ts = int(pkt.time)
                    # 获取包大小 (总长度)
                    size = len(pkt)
                    # 获取源 IP
                    src_ip = pkt[IP].src
                    
                    # === 核心逻辑：检测时间窗口切换 ===
                    if ts != current_second:
                        # 如果不是第一个包，且时间变了，说明上一秒结束了
                        if current_second != -1:
                            # 1. 计算上一秒的特征
                            rate = len(temp_sizes)
                            
                            # 只有当速率大于0时才计算统计特征
                            if rate > 0:
                                size_std = np.std(temp_sizes)
                                ip_ent = calculate_entropy(temp_ips)
                                
                                features_list.append({
                                    'timestamp': current_second,
                                    'Rate': rate,
                                    'Size_Std': size_std,
                                    'Entropy': ip_ent,
                                    'Label': label
                                })
                                processed_seconds += 1
                                
                                # 检查是否达到最大处理秒数限制
                                if max_seconds and processed_seconds >= max_seconds:
                                    print(f"\n[!] 已达到设定的最大秒数限制 ({max_seconds}s)，停止处理。")
                                    break
                        
                        # 2. 重置缓存，开始记录新的一秒
                        current_second = ts
                        temp_sizes = []
                        temp_ips = []
                    
                    # 3. 将当前包加入缓存
                    temp_sizes.append(size)
                    temp_ips.append(src_ip)
                    
            # 循环结束后，处理最后一秒的数据
            if temp_sizes and (max_seconds is None or processed_seconds < max_seconds):
                rate = len(temp_sizes)
                size_std = np.std(temp_sizes)
                ip_ent = calculate_entropy(temp_ips)
                features_list.append({
                    'timestamp': current_second,
                    'Rate': rate,
                    'Size_Std': size_std,
                    'Entropy': ip_ent,
                    'Label': label
                })

    except Exception as e:
        print(f"\n[!] 处理过程中发生错误: {e}")
        return

    print(f"\n[*] 基础特征提取完成！正在计算第4个特征(加速度)...")
    
    # 转换为 DataFrame
    if features_list:
        df = pd.DataFrame(features_list)
        
        # ======================================================
        # 新增代码：在这里直接计算第 4 个特征 (Accel)
        # ======================================================
        # Accel = 当前速率 - 上一秒速率 (使用 diff 函数)
        # fillna(0) 是因为第一行没有上一秒，结果会是 NaN，需要填 0
        df['Accel'] = df['Rate'].diff().fillna(0)
        
        # 调整列顺序，把 Accel 加进去
        cols = ['timestamp', 'Rate', 'Size_Std', 'Entropy', 'Accel', 'Label']
        df = df[cols]
        
        # 保存 CSV
        df.to_csv(output_path, index=False)
        print(f"[*] CSV 文件已保存至: {os.path.abspath(output_path)}")
        print(f"[*] 包含特征: Rate, Size_Std, Entropy, Accel (共 {len(df)} 行)")
        print("\n数据预览 (前 5 行):")
        print(df.head())
    else:
        print("[!] 没有提取到任何特征，请检查 PCAP 文件是否包含 IP 数据包。")

if __name__ == "__main__":
    extract_features(INPUT_PCAP, OUTPUT_CSV, LABEL, MAX_SECONDS)