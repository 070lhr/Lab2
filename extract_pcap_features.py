#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scapy.all import PcapReader, IP
from scipy.stats import entropy
from collections import Counter
import os
import glob
import sys
from sklearn.utils import shuffle

# ================= 配置区域 =================
# 输入：包含所有 DDoS PCAP 的目录
INPUT_DIR = '/data/exp/hrliu/CIC2023/pcap/'

# 输出：最终合并的大 CSV 文件名
OUTPUT_CSV = './ciciot_ddos_all_mixed.csv'

# 标签：DDoS 攻击标记为 1
LABEL = 1 

# 削弱策略配置
# 目标是将一部分数据的 Rate 降到这个区间，以模拟 Low-Rate DDoS
TARGET_RATE_MIN = 2000
TARGET_RATE_MAX = 5000
# ===========================================

def calculate_entropy(ip_list):
    """ 计算 IP 列表的香农熵 """
    if not ip_list:
        return 0
    counts = np.array(list(Counter(ip_list).values()))
    probs = counts / len(ip_list)
    return entropy(probs, base=2)

def extract_features_from_pcap(pcap_path):
    """ 处理单个 PCAP 文件，返回 DataFrame """
    filename = os.path.basename(pcap_path)
    print(f"[*] 正在处理文件: {filename} ...")
    
    features_list = []
    
    # 临时缓存
    current_second = -1
    temp_sizes = []
    temp_ips = []
    
    packet_count = 0
    
    try:
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                packet_count += 1
                if packet_count % 50000 == 0:
                    sys.stdout.write(f"\r    > 已扫描数据包: {packet_count} ...")
                    sys.stdout.flush()
                
                if IP in pkt:
                    ts = int(pkt.time)
                    size = len(pkt)
                    src_ip = pkt[IP].src
                    
                    # === 时间窗口切换 ===
                    if ts != current_second:
                        if current_second != -1:
                            rate = len(temp_sizes)
                            if rate > 0:
                                features_list.append({
                                    'timestamp': current_second,
                                    'Rate': rate,
                                    'Size_Std': np.std(temp_sizes),
                                    'Entropy': calculate_entropy(temp_ips),
                                    'Label': LABEL,
                                    'Source_File': filename # 记录来源，方便排查
                                })
                        
                        current_second = ts
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
    
    if not features_list:
        return None

    # 转 DataFrame
    df = pd.DataFrame(features_list)
    
    # === 关键步骤：独立计算加速度 ===
    # 必须在合并前计算，保证物理意义正确
    df = df.sort_values('timestamp')
    df['Accel'] = df['Rate'].diff().fillna(0)
    
    return df

def generate_stealthy_samples(df_original):
    """ 生成削弱版的隐蔽攻击样本 """
    print("    > 正在生成隐蔽样本 (Low-Rate Data Augmentation)...")
    
    df_stealthy = df_original.copy()
    
    # 计算当前平均速率
    current_mean_rate = df_original['Rate'].mean()
    
    # 如果原始速率本身就很低（比如本来就是 Slowloris），就不削弱了
    if current_mean_rate < TARGET_RATE_MAX:
        print("      - 原始速率较低，跳过削弱。")
        return pd.DataFrame() # 返回空
    
    # 计算缩放倍数
    # 我们希望 Rate 降到 2000~5000 之间
    # 比如原始是 14000，目标是 3500，那么 factor = 4
    # 我们生成一个随机缩放因子数组，增加多样性
    
    # 估算中心缩放因子
    center_factor = current_mean_rate / ((TARGET_RATE_MIN + TARGET_RATE_MAX) / 2)
    
    # 生成随机因子 (在中心因子周围波动)
    # 比如 center=4, 生成 3.0 ~ 5.0 之间的随机数
    factors = np.random.uniform(low=max(1.1, center_factor - 1.5), 
                                high=center_factor + 1.5, 
                                size=len(df_stealthy))
    
    # 执行削弱
    df_stealthy['Rate'] = df_stealthy['Rate'] / factors
    df_stealthy['Accel'] = df_stealthy['Accel'] / factors # 加速度也要符合物理规律地变小
    
    # 打上一个标记 (可选，或者直接混合)
    # df_stealthy['Is_Stealthy'] = True
    
    print(f"      - 生成了 {len(df_stealthy)} 条隐蔽样本，平均速率降至: {df_stealthy['Rate'].mean():.2f}")
    
    return df_stealthy

def main():
    # 1. 获取所有 pcap 文件
    pcap_files = glob.glob(os.path.join(INPUT_DIR, "*.pcap"))
    pcap_files.sort()
    
    if not pcap_files:
        print(f"[!] 错误: 在 {INPUT_DIR} 下未找到任何 .pcap 文件！")
        return

    all_dfs = []
    
    print(f"[*] 找到 {len(pcap_files)} 个 PCAP 文件，开始批量处理...")
    print("="*60)

    # 2. 循环处理每个文件
    for pcap_path in pcap_files:
        # A. 提取原始高强度特征
        df_high = extract_features_from_pcap(pcap_path)
        
        if df_high is not None and not df_high.empty:
            # B. 生成低强度隐蔽特征 ("削弱"操作)
            df_low = generate_stealthy_samples(df_high)
            
            # C. 将两者放入总列表
            all_dfs.append(df_high)
            if not df_low.empty:
                all_dfs.append(df_low)
        
        print("-" * 60)

    # 3. 合并所有数据
    print("[*] 正在合并所有数据集...")
    if not all_dfs:
        print("[!] 没有提取到任何有效数据。")
        return

    df_final = pd.concat(all_dfs, ignore_index=True)
    
    # 4. 打乱顺序 (Shuffle)
    # 对于机器学习训练，打乱是必须的，防止模型记住房文件顺序
    df_final = shuffle(df_final, random_state=42)
    
    # 5. 整理列顺序
    cols = ['timestamp', 'Rate', 'Size_Std', 'Entropy', 'Accel', 'Label', 'Source_File']
    # 如果有些列没有(比如Source_File在生成时可能丢了)，这里做个兼容
    final_cols = [c for c in cols if c in df_final.columns]
    df_final = df_final[final_cols]

    # 6. 保存
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print("="*60)
    print(f"[*] 任务完成！")
    print(f"[*] 最终文件: {os.path.abspath(OUTPUT_CSV)}")
    print(f"[*] 总样本数: {len(df_final)}")
    print(f"[*] 包含数据: 原始强攻击 (Rate ~1.5w) + 隐蔽弱攻击 (Rate ~3k)")
    print("\n数据预览:")
    print(df_final.head())
    print("\n速率统计:")
    print(df_final['Rate'].describe())

if __name__ == "__main__":
    main()