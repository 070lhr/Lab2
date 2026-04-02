import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from matplotlib.font_manager import FontProperties

# ================= 配置 =================
FLASH_FILE = './flash_event_9dim_full.csv'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
OUTPUT_IMG = 'all_9_features_distribution_full.png'

# 指定微软雅黑字体文件路径及字号
my_font = FontProperties(fname='./MSYH.TTC', size=12)
# =======================================

def plot_all_features():
    print(f"[*] 正在读取文件...")
    if not os.path.exists(FLASH_FILE) or not os.path.exists(DDOS_FILE):
        print(f"[!] 错误: 找不到文件。")
        return

    df_flash = pd.read_csv(FLASH_FILE)
    df_ddos = pd.read_csv(DDOS_FILE)
    
    # 强制指定特征的绘制顺序
    features_to_check = [
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA',  
        'SIP_Ent', 'SIPEnt_MA', 'SIPEnt_Change',      
        'Rate', 'Rate_Accel', 'Rate_CV'              
    ]
    
    n_features = len(features_to_check)
    
    cols = 3
    rows = math.ceil(n_features / cols)
    
    # 动态调整高度，稍微收紧画布总高度
    plt.figure(figsize=(18, 4.0 * rows)) 
    
    for i, col in enumerate(features_to_check):
        ax = plt.subplot(rows, cols, i+1)
        
        # ================= 核心优化区 =================
        # 合并两类流量，计算全局的 1% 和 99% 分位数，统一量纲边界
        combined_data = pd.concat([df_flash[col], df_ddos[col]])
        lower_bound = combined_data.quantile(0.01)
        upper_bound = combined_data.quantile(0.99)
        
        # 剔除极端离群点，提取画图专用数据
        plot_flash = df_flash[(df_flash[col] >= lower_bound) & (df_flash[col] <= upper_bound)]
        plot_ddos = df_ddos[(df_ddos[col] >= lower_bound) & (df_ddos[col] <= upper_bound)]
        # ==============================================
        
        # 绘制 FE (蓝色) - 加入 kde_kws={'bw_adjust': 1.5} 提升曲线平滑度
        sns.histplot(plot_flash[col], color='blue', label='FE', 
                     kde=True, stat="percent", common_norm=False, 
                     element="step", fill=True, alpha=0.3,
                     kde_kws={'bw_adjust': 1.5})
        
        # 绘制 DDoS (红色)
        sns.histplot(plot_ddos[col], color='red', label='DDoS', 
                     kde=True, stat="percent", common_norm=False, 
                     element="step", fill=True, alpha=0.3,
                     kde_kws={'bw_adjust': 1.5})
        
        subplot_label = chr(97 + i)  # 生成 a, b, c...
        
        # 1. 横坐标保持原样
        plt.xlabel(col, fontsize=12)
        
        # 2. 把 -0.22 向上收紧到了 -0.15，让标号紧紧贴着横坐标
        plt.text(0.5, -0.17, f'({subplot_label})', 
                 transform=ax.transAxes, 
                 fontsize=14, 
                 ha='center', va='top')
        
        plt.ylabel('样本占比(%)', fontproperties=my_font)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 如果某些变化率特征裁剪后依然跨度极大且密集在0，取消下面两行的注释开启对称对数坐标
        # if col in ['SizeStd_Change', 'SIPEnt_Change', 'Rate_Accel']:
        #     ax.set_xscale('symlog')

    # 把 h_pad 从 3.0 收紧到了 1.5，让排与排之间的空白恰到好处
    plt.tight_layout(h_pad=1.5) 
    
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 绘图完成！优化后的图片已保存为: {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_all_features()