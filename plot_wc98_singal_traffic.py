#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# ================= 配置区域 =================
# 输入文件路径 (请确认路径是否正确)
INPUT_CSV = '/data/exp/hrliu/WC98/wc_day73_1.csv'

# 输出图片文件名
OUTPUT_IMG = './wc98_day73_traffic_rate.png'

# 筛选阈值线 (可选): 在图中画一条红线，表示您筛选 Flash Event 的门槛
# 例如：只看每秒请求数 > 2000 的部分
THRESHOLD = 2000 
# ===========================================

def plot_traffic_rate(input_path, output_path):
    print(f"[*] 正在读取数据: {input_path} ...")
    
    if not os.path.exists(input_path):
        print(f"[!] 错误: 找不到文件 {input_path}")
        return

    try:
        # 只读取 timestamp 列，速度最快且省内存
        # 如果您的 CSV 没有表头，pandas 可能会把第一行当表头，
        # 所以保险起见，我们尝试手动指定列名读取，或者假设有表头
        # 这里使用一种通用策略：先读几行看看
        df_test = pd.read_csv(input_path, nrows=5)
        if 'timestamp' in df_test.columns:
            df = pd.read_csv(input_path, usecols=['timestamp'])
        else:
            # 假设没有表头，第一列是 timestamp
            print("[*] 未检测到表头，假设第一列为时间戳...")
            df = pd.read_csv(input_path, usecols=[0], names=['timestamp'], header=None)
            
    except Exception as e:
        print(f"[!] 读取失败: {e}")
        return

    print(f"[*] 数据加载完成，共 {len(df)} 条记录。正在计算每秒速率...")

    # === 核心计算：每秒有多少行 ===
    # value_counts() 比 groupby().size() 通常要快一点
    rate_series = df['timestamp'].value_counts().sort_index()
    
    # 转换为 DataFrame 方便绘图
    rate_df = rate_series.reset_index()
    rate_df.columns = ['timestamp', 'requests_per_second']
    
    # 将时间戳转换为相对时间 (从第0秒开始)，这样图表更好看
    # 或者保留原始时间戳，看您需求。这里用相对时间展示“一天内的趋势”
    start_time = rate_df['timestamp'].min()
    rate_df['relative_time'] = rate_df['timestamp'] - start_time
    
    print("[*] 正在绘图...")
    
    # 设置画布风格
    plt.figure(figsize=(12, 6), dpi=300)
    
    # 绘制折线图
    # 颜色选用 'royalblue' (宝蓝色)，学术论文常用色
    plt.plot(rate_df['relative_time'], rate_df['requests_per_second'], 
             color='royalblue', linewidth=0.8, label='Traffic Rate')

    # === 可视化筛选阈值 (可选) ===
    if THRESHOLD > 0:
        plt.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=1.5, 
                    label=f'Flash Event Threshold ({THRESHOLD} req/s)')
        
        # 统计一下有多少点超过了阈值
        peak_points = rate_df[rate_df['requests_per_second'] > THRESHOLD]
        print(f"[*] 统计: 共有 {len(peak_points)} 秒的流量超过了阈值 {THRESHOLD}。")

    # 设置标签和标题
    plt.title('World Cup 98 (Day 73) Traffic Rate Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Seconds from start)', fontsize=12)
    plt.ylabel('Requests / Second', fontsize=12)
    
    # 设置网格
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    # 优化刻度显示
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path)
    print(f"[*] 绘图完成！图片已保存至: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    plot_traffic_rate(INPUT_CSV, OUTPUT_IMG)