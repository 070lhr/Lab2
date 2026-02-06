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
OUTPUT_CSV = './ciciot_ddos_9dim_final.csv'

# 标签：DDoS = 1
LABEL = 1 

# 降维打击参数：将高频攻击削弱到的目标区间
TARGET_RATE_MIN = 2000
TARGET_RATE_MAX = 5000
# ===========================================

def calculate_entropy(ip_list):
    """ 计算 IP 列表的香农熵 """
    if not ip_list: return 0
    counts = np.array(list(Counter(ip_list).values()))
    probs = counts / len(ip_list)
    return entropy(probs, base=2)

def compute_rolling_features(df):
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
    # f3: 速率波动率 (5秒窗口标准差，反映流量是否"死板")
    df['Rate_Vol'] = df['Rate'].rolling(window=5, min_periods=1).std().fillna(0)
    
    # === 2. 熵维度 (Entropy Dimension) ===
    # f4: Entropy (基础值)
    # f5: 熵的变化率 (攻击开始/结束时的突变)
    df['Ent_Change'] = df['Entropy'].diff().fillna(0)
    # f6: 熵的移动平均 (5秒均值，消除抖动看长期趋势)
    df['Ent_MA'] = df['Entropy'].rolling(window=5, min_periods=1).mean().fillna(df['Entropy'])
    
    # === 3. 载荷维度 (Payload Dimension) ===
    # f7: Size_Std (基础值)
    # f8: 载荷标准差的变化
    df['SizeStd_Change'] = df['Size_Std'].diff().fillna(0)
    # f9: 载荷标准差的均值 (5秒均值)
    df['SizeStd_MA'] = df['Size_Std'].rolling(window=5, min_periods=1).mean().fillna(df['Size_Std'])
    
    # 清理因 diff 产生的 NaN (填充为 0)
    df = df.fillna(0)
    
    return df

def apply_stealthy_mix_9dim(df_original):
    """ 
    【降维打击策略】
    随机选择 50% 的数据，将其速率相关特征削弱，模拟 Low-Rate DDoS。
    另外 50% 保持原始高强度。
    """
    if df_original is None or df_original.empty: 
        return pd.DataFrame()
    
    current_mean_rate = df_original['Rate'].mean()
    
    # 如果原始速率本身就很低，就不削弱了，直接返回
    if current_mean_rate < TARGET_RATE_MAX: 
        return df_original
    
    # 1. 随机打乱并重置索引
    df_shuffled = df_original.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. 找到切分点 (50%)
    split_point = len(df_shuffled) // 2
    
    # 3. 切分数据
    df_keep_high = df_shuffled.iloc[:split_point].copy() # 保留原始
    df_to_modify = df_shuffled.iloc[split_point:].copy() # 准备削弱
    
    # 4. 计算削弱因子 (随机浮动)
    center_factor = current_mean_rate / ((TARGET_RATE_MIN + TARGET_RATE_MAX) / 2)
    # 生成随机因子数组，范围在中心因子周围 ±1.5
    factors = np.random.uniform(low=max(1.1, center_factor - 1.5), 
                                high=center_factor + 1.5, 
                                size=len(df_to_modify))
    
    # 5. 执行削弱 (只削弱与"量"有关的特征)
    # Rate 变小
    df_to_modify['Rate'] = df_to_modify['Rate'] / factors
    
    # 衍生特征也要变：加速度变小
    df_to_modify['Rate_Accel'] = df_to_modify['Rate_Accel'] / factors
    # 波动率变小 (标准差随幅度下降)
    df_to_modify['Rate_Vol'] = df_to_modify['Rate_Vol'] / factors
    
    # 【注意】熵 (Entropy) 和包大小 (Size_Std) 系列特征不需要除以 factor
    # 因为攻击变慢了，不代表它的 IP 分布变了，也不代表包大小特征变了
    
    # 6. 合并回去
    df_mixed = pd.concat([df_keep_high, df_to_modify], ignore_index=True)
    
    return df_mixed

def process_single_pcap(pcap_path):
    """
    工作进程：处理单个 PCAP 的完整流程
    1. 解析 PCAP -> 2. 按秒聚合 -> 3. 计算9特征 -> 4. 混合削弱
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
                                'Entropy': calculate_entropy(temp_ips), # f4 基础
                                'Label': LABEL,
                                # 其他特征稍后通过 compute_rolling_features 计算
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
                    'Entropy': calculate_entropy(temp_ips),
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
    
    # 2. 计算 9 个高级时序特征
    # (这一步会自动排序并计算 diff/rolling)
    df = compute_rolling_features(df)
    
    # 3. 应用 50% 混合削弱策略 (降维打击)
    df_mixed = apply_stealthy_mix_9dim(df)
    
    print(f"[PID {pid}] 完成 {filename}: 共 {len(df_mixed)} 条样本")
    
    # 返回列表 (multiprocessing 要求)
    return [df_mixed]

def main():
    # 1. 扫描文件
    pcap_files = glob.glob(os.path.join(INPUT_DIR, "*.pcap"))
    
    if not pcap_files:
        print(f"[!] 错误: 在 {INPUT_DIR} 下未找到任何 .pcap 文件！")
        return

    # 2. 自动检测 CPU 核数
    max_workers = os.cpu_count() or 4
    print(f"[*] 检测到 {max_workers} 个 CPU 核心，开始并行处理 {len(pcap_files)} 个文件...")
    print(f"[*] 目标: 提取 9 维特征 + 50% 替换为隐蔽样本")
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
    
    # 6. 整理列顺序 (只保留训练需要的列)
    # 定义 9 个特征列名
    feature_cols = [
        'Rate', 'Rate_Accel', 'Rate_Vol', 
        'Entropy', 'Ent_Change', 'Ent_MA', 
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA', 
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