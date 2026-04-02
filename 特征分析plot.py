import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from matplotlib.font_manager import FontProperties

# ================= 配置 =================
FLASH_FILE = './flash_event_9dim_full.csv'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
OUTPUT_IMG = 'all_9_features_distribution_full_percent.png'

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
    
    features_to_check = [
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA',  
        'SIP_Ent', 'SIPEnt_MA', 'SIPEnt_Change',      
        'Rate', 'Rate_Accel', 'Rate_CV'              
    ]
    
    n_features = len(features_to_check)
    cols = 3
    rows = math.ceil(n_features / cols)
    
    plt.figure(figsize=(18, 4.0 * rows)) 
    
    for i, col in enumerate(features_to_check):
        ax = plt.subplot(rows, cols, i+1)
        
        # ================= 核心优化区 =================
        # 1. 使用 stat='percent' 确保 Y 轴是真实的百分比数值
        # 2. 使用 element='poly' 和 bins=100 让直方图变成平滑的填充曲线
        sns.histplot(df_flash[col], color='blue', label='FE', 
                     stat='percent', element='poly', fill=True, alpha=0.3, bins=100, ax=ax)
        
        sns.histplot(df_ddos[col], color='red', label='DDoS', 
                     stat='percent', element='poly', fill=True, alpha=0.3, bins=100, ax=ax)
        
        # 3. 计算联合数据的视觉边界（放宽到 0.1% 和 99.9%）
        combined_data = pd.concat([df_flash[col], df_ddos[col]])
        q_low = combined_data.quantile(0.001)
        q_high = combined_data.quantile(0.999)
        
        # 4. 使用坐标轴缩放 (Zoom-in) 隐藏极端的长尾，避免截断的悬崖感
        margin = (q_high - q_low) * 0.05 if q_high != q_low else 1
        ax.set_xlim(q_low - margin, q_high + margin)
        # ==============================================
        
        subplot_label = chr(97 + i)
        
        plt.xlabel(col, fontsize=12)
        plt.text(0.5, -0.17, f'({subplot_label})', 
                 transform=ax.transAxes, 
                 fontsize=14, 
                 ha='center', va='top')
        
        plt.ylabel('样本占比(%)', fontproperties=my_font)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(h_pad=1.5) 
    
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 绘图完成！完美百分比图片已保存为: {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_all_features()