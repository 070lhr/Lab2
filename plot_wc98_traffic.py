import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os

# ================= 配置区域 =================
# CSV 文件所在的目录
INPUT_DIR = './WorldCupCSV'

# 图片保存路径
OUTPUT_IMAGE = 'wc98_traffic_analysis.png'
# ===========================================

def load_and_aggregate():
    # 找到所有 CSV 文件
    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    
    if not all_files:
        print(f"错误：在 {INPUT_DIR} 下未找到任何 .csv 文件！")
        return None

    print(f"找到 {len(all_files)} 个 CSV 文件，开始处理...")
    
    aggregated_chunks = []

    for i, file in enumerate(all_files):
        print(f"[{i+1}/{len(all_files)}] 正在读取并聚合: {os.path.basename(file)} ...", end='\r')
        
        try:
            # 只读取需要的列以节省内存
            # timestamp: 用于分组
            # size: 用于计算流量和请求数(count)
            df = pd.read_csv(file, usecols=['timestamp', 'size'])
            
            # 按秒聚合
            # count: 这一秒有多少行 = 请求数
            # sum: 这一秒 size 总和 = 流量
            df_agg = df.groupby('timestamp')['size'].agg(['count', 'sum']).reset_index()
            df_agg.rename(columns={'count': 'request_count', 'sum': 'traffic_bytes'}, inplace=True)
            
            aggregated_chunks.append(df_agg)
            
        except Exception as e:
            print(f"\n[跳过] 文件 {file} 读取失败: {e}")

    print("\n正在合并所有数据...")
    
    if not aggregated_chunks:
        print("没有有效的数据被加载。")
        return None

    # 合并所有小文件的聚合结果
    full_df = pd.concat(aggregated_chunks, ignore_index=True)
    
    # 再次聚合（防止不同文件包含同一秒的数据）
    final_df = full_df.groupby('timestamp').sum().reset_index()
    
    # 将 Unix 时间戳转换为 datetime 对象
    final_df['datetime'] = pd.to_datetime(final_df['timestamp'], unit='s')
    
    # 按时间排序
    final_df.sort_values('timestamp', inplace=True)
    
    return final_df

def plot_traffic(df):
    print("正在绘图...")
    
    # 设置画布大小
    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    # === 绘制左轴：请求数 (Requests/s) ===
    color = 'tab:blue'
    ax1.set_xlabel('Time (Date Hour)', fontsize=12)
    ax1.set_ylabel('Request Rate (Reqs/sec)', color=color, fontsize=12)
    ax1.plot(df['datetime'], df['request_count'], color=color, linewidth=1, label='Request Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # === 绘制右轴：流量 (MB/s) ===
    ax2 = ax1.twinx()  # 共享x轴
    color = 'tab:orange'
    # 将字节转换为 MB
    traffic_mb = df['traffic_bytes'] / (1024 * 1024)
    ax2.set_ylabel('Traffic Rate (MB/sec)', color=color, fontsize=12)
    ax2.plot(df['datetime'], traffic_mb, color=color, linewidth=1, alpha=0.7, label='Traffic Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    # === 设置标题和格式 ===
    plt.title('World Cup 98 Traffic Analysis (Requests vs Traffic)', fontsize=16)
    
    # 格式化 X 轴时间显示 (例如: 06-10 12:00)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate() # 自动旋转日期标签

    # 保存图片
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"绘图完成！图片已保存为: {OUTPUT_IMAGE}")
    
    # 如果你在本地运行，可以取消下面这行的注释来显示图片
    # plt.show()

if __name__ == "__main__":
    df = load_and_aggregate()
    
    if df is not None and not df.empty:
        print(f"总样本数 (秒): {len(df)}")
        print("数据预览:")
        print(df.head())
        plot_traffic(df)
    else:
        print("数据为空，无法绘图。")