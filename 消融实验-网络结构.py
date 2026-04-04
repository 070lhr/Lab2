import matplotlib
matplotlib.use('Agg') # 后台静默出图
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import os

# ==================== 1. 字体配置 (参考文献做法) ====================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
font_en_path = os.path.join(current_dir, "TIMES.TTF")
font_cn_path = os.path.join(current_dir, "SIMSUN.TTC")

if not os.path.exists(font_en_path) or not os.path.exists(font_cn_path):
    print("警告：请确保 TIMES.TTF 和 SIMSUN.TTC 文件在当前执行目录下！")

# 创建字体属性对象
font_en = font_manager.FontProperties(fname=font_en_path, size=18)
font_cn = font_manager.FontProperties(fname=font_cn_path, size=18)
font_legend = font_manager.FontProperties(fname=font_cn_path, size=16) # 图例含中文，用宋体
font_ticks = font_manager.FontProperties(fname=font_en_path, size=18)

# 全局配置
plt.rcParams['axes.unicode_minus'] = False 

# ==================== 2. 数据准备 ====================
labels = ['Dist-Only', 'Dyn-Only', 'DPDADFE']
x = np.arange(len(labels))
width = 0.32  

# 数据
acc_clean = [0.8920, 0.8750, 0.9942] 
acc_adv   = [0.4950, 0.8450, 0.8880] 
pre_clean = [0.8750, 0.8520, 0.9905]
pre_adv   = [0.1250, 0.8450, 0.9180] 
rec_clean = [0.9150, 0.8380, 0.9925]
rec_adv   = [0.0480, 0.7780, 0.7880]
f1_clean  = [0.8946, 0.8449, 0.9915]
f1_adv    = [0.0694, 0.8101, 0.8481]

all_data_clean = [acc_clean, pre_clean, rec_clean, f1_clean]
all_data_adv = [acc_adv, pre_adv, rec_adv, f1_adv]
titles_cn = ['准确率', '精确率', '召回率', 'F1分数']

# ==================== 3. 绘图执行 ====================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

def plot_bar_subplot(ax, data_clean, data_adv, ylabel_cn, i):
    # 绘制柱状图
    ax.bar(x - width/2, data_clean, width, label='正常环境', 
           color='#5B8FF9', edgecolor='black', hatch='//', alpha=0.9)
    ax.bar(x + width/2, data_adv, width, label='对抗攻击', 
           color='#E8684A', edgecolor='black', hatch='\\\\', alpha=0.9)
    
    # 坐标轴限制
    ax.set_ylim(0, 1.0) 
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # X轴标签 (算法名使用 Times New Roman)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_ticks)
    
    # Y轴刻度
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_ticks)

    # Y轴名称 (宋体)
    ax.set_ylabel(ylabel_cn, fontproperties=font_cn, fontsize=18)
    
    # ================= 核心拆分：混合字体标题 (a) 准确率 =================
    # 清空默认 xlabel
    ax.set_xlabel('', labelpad=25)
    
    letter = f'({chr(97+i)})' # (a), (b)...
    # 字母用 Times，靠左；中文用宋体，靠右，拼接在中间
    ax.text(0.46, -0.15, letter, transform=ax.transAxes, fontproperties=font_en, ha='right', va='center')
    ax.text(0.48, -0.15, titles_cn[i], transform=ax.transAxes, fontproperties=font_cn, ha='left', va='center')
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# 循环绘制
for i in range(4):
    plot_bar_subplot(axs[i], all_data_clean[i], all_data_adv[i], titles_cn[i], i)

# ============ 布局与图例 ============
# 增大 wspace 为中间图例留出足够空间
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.12, hspace=0.35, wspace=0.25)

handles, labels_legend = axs[0].get_legend_handles_labels()

# 图例使用宋体 (因为含中文)
fig.legend(handles, labels_legend, loc='center', ncol=1, prop=font_legend, 
           frameon=True, edgecolor='black', facecolor='white', framealpha=1,
           bbox_to_anchor=(0.51, 0.527))

OUTPUT_IMG = '消融实验-网络结构图.png'
plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
print(f"\n[+] 字体修改成功！字母使用 Times New Roman，中文使用宋体。请查看: {OUTPUT_IMG}")