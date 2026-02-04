#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import glob
import re
import sys

# ================= 配置区域 =================
# 输入：包含所有 CSV 文件的目录路径
INPUT_DIR = '/data/exp/hrliu/1998WC/WorldCupCSV/'

# 输出：图片保存路径
OUTPUT_IMG = './wc98_day71_74_traffic.png'

# 筛选范围：只保留这几天的数据
TARGET_DAYS = [71, 72, 73, 74]
# ===========================================

def get_sorted_target_files(directory, target_days):
    """
    获取目录下指定天数(target_days)的csv文件，并按文件名排序。
    """
    files = glob.glob(os.path.join(directory, "*.csv"))
    target_files = []
    
    # 定义正则提取 day 和 part
    pattern = re.compile(r'day(\d+)_(\d+)')

    for f in files:
        filename = os.path.basename(f)
        match = pattern.search(filename)
        if match:
            day = int(match.group(1))
            part = int(match.group(2))
            
            # 核心筛选：只保留 target_days 里的天数
            if day in target_days:
                target_files.append((day, part, f))
    
    # 按 (day, part) 排序
    target_files.sort(key=lambda x: (x[0], x[1]))
    
    # 只返回文件路径
    return [x[2] for x in target_files]

def plot_specific_days(input_dir, output_img):
    print(f"[*] 正在扫描目录: {input_dir}")
    print(f"[*] 目标天数: {TARGET_DAYS}")
    
    # 1. 获取筛选后的文件列表
    files_to_process = get_sorted_target_files(input_dir, TARGET_DAYS)
    
    if not files_to_process:
        print(f"[!] 错误: 未找到 Day {TARGET_DAYS} 的任何 .csv 文件！")
        return

    print(f"[*] 找到 {len(files_to_process)} 个相关文件，开始处理...")
    
    daily_stats_list = []
    total_requests = 0

    # 2. 循环读取并聚合
    for i, file_path in enumerate(files_to_process):
        file_name = os.path.basename(file_path)
        sys.stdout.write(f"\r[{i+1}/{len(files_to_process)}] 正在处理: {file_name} ...")
        sys.stdout.flush()
        
        try:
            # 尝试读取 timestamp 列
            try:
                # 预读一行检测表头
                df_test = pd.read_csv(file_path, nrows=1)
                if 'timestamp' in df_test.columns:
                    df = pd.read_csv(file_path, usecols=['timestamp'])
                else:
                    # 无表头，假设第一列是时间戳
                    df = pd.read_csv(file_path, usecols=[0], names=['timestamp'], header=None)
            except:
                # 兜底
                df = pd.read_csv(file_path, usecols=[0], names=['timestamp'], header=None)

            # 计算每秒请求数
            counts = df['timestamp'].value_counts().sort_index()
            daily_stats_list.append(counts)
            total_requests += len(df)
            
            del df

        except Exception as e:
            print(f"\n[!] 读取文件 {file_name} 失败: {e}")

    print(f"\n[*] 读取完毕，正在合并数据...")

    # 3. 合并与绘图
    if not daily_stats_list:
        return

    full_series = pd.concat(daily_stats_list)
    # 按时间戳合并重叠部分
    full_series = full_series.groupby(level=0).sum().sort_index()
    
    df_plot = full_series.reset_index()
    df_plot.columns = ['timestamp', 'count']
    df_plot['datetime'] = pd.to_datetime(df_plot['timestamp'], unit='s')

    print("[*] 正在绘图...")

    plt.figure(figsize=(15, 6), dpi=300) # 宽图
    
    # 绘制折线
    plt.plot(df_plot['datetime'], df_plot['count'], color='#1f77b4', linewidth=0.8, label='Request Rate')
    
    plt.title(f'World Cup 98 Traffic Rate (Day {min(TARGET_DAYS)} - Day {max(TARGET_DAYS)})', fontsize=16, fontweight='bold')
    plt.xlabel('Date / Time', fontsize=12)
    plt.ylabel('Requests / Second', fontsize=12)
    
    # X轴格式化：精确到小时
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6)) # 每6小时一个刻度
    plt.gcf().autofmt_xdate()

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 标记出这几天的最高峰
    max_rate = df_plot['count'].max()
    max_time = df_plot.loc[df_plot['count'].idxmax(), 'datetime']
    plt.annotate(f'Peak: {max_rate}\n({max_time.strftime("%d %H:%M")})', 
                 xy=(max_time, max_rate), 
                 xytext=(max_time, max_rate + 200),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"[*] 绘图完成！图片已保存至: {os.path.abspath(output_img)}")

if __name__ == "__main__":
    plot_specific_days(INPUT_DIR, OUTPUT_IMG)