#!/usr/bin/env python3
from matplotlib.font_manager import FontProperties
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
OUTPUT_IMG = './98WC数据示意图.png'

# 筛选范围：只保留这几天的数据
TARGET_DAYS = [72, 73, 74, 75]

# 设置中文字体（请确保系统中有该字体，若报错可去掉 fontproperties 参数）
my_font = FontProperties(fname='./MSYH.TTC', size=12) 
# 如果找不到该路径的字体，可以回退使用默认设置：
# my_font = FontProperties(size=12) 
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

    print(f"[*] 找到 {len(files_to_process)} 个相关文件，开始处理...")
    
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
            
            # 存入列表
            short_name = file_name.replace('.csv', '')
            data_segments.append((short_name, seg_df))
            
            del df

        except Exception as e:
            print(f"\n[!] 读取文件 {file_name} 失败: {e}")

    print(f"\n[*] 读取完毕，开始绘图...")

    if not data_segments:
        return

    # 2. 绘图设置
    plt.figure(figsize=(18, 8), dpi=300) 
    
    # 设定统一的颜色
    main_plot_color = 'steelblue' 
    max_y_val = 0 

    # 3. 循环绘制每一段 (彻底取消了分隔线和标签标注)
    for idx, (fname, df) in enumerate(data_segments):
        
        # 绘制纯净曲线
        plt.plot(df['datetime'], df['count'], color=main_plot_color, linewidth=1)
        
        # 更新最大Y值用于后面设置范围
        current_max = df['count'].max()
        if current_max > max_y_val:
            max_y_val = current_max

    # 4. 图表美化
    plt.xlabel('时间', fontproperties=my_font)
    plt.ylabel('每秒请求次数', fontproperties=my_font)
    
    # Y轴从0开始，顶部留出 5% 的空间即可
    plt.ylim(0, max_y_val * 1.05)
    
    # X轴格式化
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.gcf().autofmt_xdate()

    # 保留虚线网格，方便看数据高度
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"[*] 绘图完成！图片已保存至: {os.path.abspath(output_img)}")

if __name__ == "__main__":
    plot_labeled_days(INPUT_DIR, OUTPUT_IMG)