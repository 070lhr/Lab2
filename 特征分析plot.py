import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from matplotlib.font_manager import FontProperties

# ================= 配置 =================
# 指定新的文件名 (保持不变)
FLASH_FILE = './flash_event_9dim_full.csv'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
OUTPUT_IMG = 'all_9_features_distribution_full.png'

# 指定微软雅黑字体文件路径及字号（确保 MSYH.TTC 与本脚本在同级目录）
my_font = FontProperties(fname='./MSYH.TTC', size=12)
# =======================================

def plot_all_features():
    print(f"[*] 正在读取文件...")
    if not os.path.exists(FLASH_FILE) or not os.path.exists(DDOS_FILE):
        print(f"[!] 错误: 找不到文件。请确认当前目录下存在:\n    {FLASH_FILE}\n    {DDOS_FILE}")
        return

    df_flash = pd.read_csv(FLASH_FILE)
    df_ddos = pd.read_csv(DDOS_FILE)
    
    # 2. 自动筛选特征列
    # 排除掉不需要画图的列 (Label, timestamp, 索引等)
    exclude_cols = ['Label', 'timestamp', 'Unnamed: 0']
    
    # 自动获取特征列表
    features_to_check = [col for col in df_flash.columns if col not in exclude_cols]
    
    n_features = len(features_to_check)
    print(f"[*] 检测到 {n_features} 个特征: {features_to_check}")
    
    if n_features == 0:
        print("[!] 未检测到特征列，请检查 CSV 文件表头。")
        return

    # 3. 设置画布布局 (自动计算行数，固定 3 列)
    cols = 3
    rows = math.ceil(n_features / cols)
    
    # 动态调整高度：每行给 4 英寸高度
    plt.figure(figsize=(18, 4 * rows))
    
    # 4. 循环绘制每个特征
    for i, col in enumerate(features_to_check):
        plt.subplot(rows, cols, i+1)
        
        # 绘制 Flash (蓝色) -> 改为 percent
        sns.histplot(df_flash[col], color='blue', label='FE', 
                     kde=True, stat="percent", common_norm=False, 
                     element="step", fill=True, alpha=0.3)
        
        # 绘制 DDoS (红色) -> 改为 percent
        sns.histplot(df_ddos[col], color='red', label='DDoS', 
                     kde=True, stat="percent", common_norm=False, 
                     element="step", fill=True, alpha=0.3)
        
        plt.title(f'{col}', fontsize=14, fontweight='bold')
        plt.xlabel(col, fontsize=12)
        
        # 修改 Y 轴标签为 Percentage (%)，并应用指定的本地中文字体
        plt.ylabel('样本占比(%)', fontproperties=my_font)
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 绘图完成！图片已保存为: {OUTPUT_IMG}")
    print("    现在 Y 轴代表‘样本占比(%)’，数值（如 20）代表该区间包含了 20% 的样本。")

if __name__ == "__main__":
    plot_all_features()