#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import os
import glob
import re
import sys

# ================= 配置区域 =================
# 输入：包含所有 CSV 文件的目录路径
INPUT_DIR = '/data/exp/hrliu/1998WC/WorldCupCSV/'

# 输出：图片保存路径
OUTPUT_IMG = './wc98_day71_74_labeled_traffic.png'

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
            
            if day in target_days:
                target_files.append((day, part, f))
    
    # 按 (day, part) 排序
    target_files.sort(key=lambda x: (x[0], x[1]))
    
    return [x[2] for x in target_files]

def plot_labeled_days(input_dir, output_img):
    print(f"[*] 正在扫描目录: {input_dir}")
    print(f"[*] 目标天数: {TARGET_DAYS}")
    
    files_to_process = get_sorted_target_files(input_dir, TARGET_DAYS)
    
    if not files_to_process:
        print(f"[!] 错误: 未找到 Day {TARGET_DAYS} 的任何 .csv 文件！")
        return

    print(f"[*] 找到 {len(files_to_process)} 个相关文件，开始分段处理...")
    
    # 用一个列表存储分段数据：[(filename, dataframe), ...]
    data_segments = []
    
    # 1. 循环读取并分别存储
    for i, file_path in enumerate(files_to_process):
        file_name = os.path.basename(file_path)
        sys.stdout.write(f"\r[{i+1}/{len(files_to_process)}] 正在处理: {file_name} ...")
        sys.stdout.flush()
        
        try:
            try:
                df_test = pd.read_csv(file_path, nrows=1)
                if 'timestamp' in df_test.columns:
                    df = pd.read_csv(file_path, usecols=['timestamp'])
                else:
                    df = pd.read_csv(file_path, usecols=[0], names=['timestamp'], header=None)
            except:
                df = pd.read_csv(file_path, usecols=[0], names=['timestamp'], header=None)

            # 计算每秒请求数
            counts = df['timestamp'].value_counts().sort_index()
            
            # 转换为 DataFrame 并添加时间列
            seg_df = counts.reset_index()
            seg_df.columns = ['timestamp', 'count']
            seg_df['datetime'] = pd.to_datetime(seg_df['timestamp'], unit='s')
            
            # 存入列表，保留文件名信息
            # 这里取简短文件名 (如 wc_day71_1) 用于标注
            short_name = file_name.replace('.csv', '')
            data_segments.append((short_name, seg_df))
            
            del df

        except Exception as e:
            print(f"\n[!] 读取文件 {file_name} 失败: {e}")

    print(f"\n[*] 读取完毕，开始分段绘图...")

    if not data_segments:
        return

    # 2. 绘图设置
    plt.figure(figsize=(18, 8), dpi=300) # 加大宽度以便显示标签
    
    # 准备一组颜色循环使用，以便区分相邻的文件
    colors = list(mcolors.TABLEAU_COLORS.values()) # 默认10种颜色循环
    
    max_y_val = 0 # 用于确定标签的高度

    # 3. 循环绘制每一段
    for idx, (fname, df) in enumerate(data_segments):
        # 获取当前颜色
        color = colors[idx % len(colors)]
        
        # 绘制曲线
        plt.plot(df['datetime'], df['count'], color=color, linewidth=1, label=None)
        
        # 更新最大Y值用于后面设置范围
        current_max = df['count'].max()
        if current_max > max_y_val:
            max_y_val = current_max
            
        # === 核心改进：标注文件边界 ===
        start_time = df['datetime'].min()
        
        # A. 画垂直虚线 (表示文件开始)
        plt.axvline(x=start_time, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # B. 标注文件名
        # 为了防止文字重叠，我们可以让高度错开一下 (交替显示)
        # 比如：偶数个文件显示在高处，奇数个文件稍微低一点
        text_y_pos = max_y_val * 1.05 if idx % 2 == 0 else max_y_val * 0.95
        
        # 截取名字中的关键部分，例如 wc_day71_1 -> 71_1
        label_text = fname.replace('wc_day', '') 
        
        plt.text(start_time, text_y_pos, label_text, 
                 rotation=90,          # 垂直旋转文字
                 verticalalignment='bottom', 
                 horizontalalignment='right',
                 fontsize=9, 
                 color='black', 
                 fontweight='bold')

    # 4. 图表美化
    plt.title(f'World Cup 98 Traffic Source Analysis (Labeled by File Segment)', fontsize=16, fontweight='bold')
    plt.xlabel('Date / Time', fontsize=12)
    plt.ylabel('Requests / Second', fontsize=12)
    
    # 扩大一点Y轴范围给标签留空间
    plt.ylim(0, max_y_val * 1.2)
    
    # X轴格式化
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.gcf().autofmt_xdate()

    plt.grid(True, linestyle=':', alpha=0.5)
    
    # 添加一个说明图例
    # 创建一个虚拟的 handle 告诉用户不同颜色代表不同文件
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors[0], lw=2),
                    Line2D([0], [0], color=colors[1], lw=2)]
    plt.legend(custom_lines, ['File Segment N', 'File Segment N+1'], loc='upper right')

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"[*] 绘图完成！图片已保存至: {os.path.abspath(output_img)}")

if __name__ == "__main__":
    plot_labeled_days(INPUT_DIR, OUTPUT_IMG)