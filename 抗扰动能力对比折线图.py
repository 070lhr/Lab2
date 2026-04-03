import matplotlib
matplotlib.use('Agg') # 后台静默出图
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import os

# ==================== 1. 字体配置 (本地文件绝对分离) ====================
# 获取当前目录下字体文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
font_en_path = os.path.join(current_dir, "TIMES.TTF")
font_cn_path = os.path.join(current_dir, "SIMSUN.TTC")

if not os.path.exists(font_en_path) or not os.path.exists(font_cn_path):
    print("警告：请确保 TIMES.TTF 和 SIMSUN.TTC 文件在当前执行目录下！")

# 分别创建纯英文和纯中文的字体属性对象
font_en = font_manager.FontProperties(fname=font_en_path, size=18)
font_cn = font_manager.FontProperties(fname=font_cn_path, size=18)
font_legend = font_manager.FontProperties(fname=font_en_path, size=14)
font_ticks = font_manager.FontProperties(fname=font_en_path, size=18)

# 劫持全局数学公式引擎（保护图中的 epsilon）
font_manager.fontManager.addfont(font_en_path)
font_en_name = font_manager.FontProperties(fname=font_en_path).get_name()
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = font_en_name
plt.rcParams['mathtext.it'] = font_en_name + ':italic'
plt.rcParams['axes.unicode_minus'] = False 

# ==================== 2. 数据准备 ====================
epsilons = [0.0, 0.2, 0.4, 0.6, 0.8]
models = ['Entropy-KL-ML', 'CCF-ZI', 'RF-RFE', 'iCuSMAT-DT', 'Lucid', 'DPDADFE']


acc_data = [
    [0.962, 0.910, 0.860, 0.710, 0.580], [0.958, 0.820, 0.650, 0.540, 0.510], 
    [0.974, 0.940, 0.880, 0.620, 0.490], [0.961, 0.890, 0.680, 0.510, 0.480], 
    [0.982, 0.680, 0.550, 0.480, 0.450], [0.994, 0.945, 0.912, 0.895, 0.888]  
]
pre_data = [
    [0.925, 0.880, 0.850, 0.720, 0.610], [0.946, 0.860, 0.710, 0.550, 0.420],
    [0.965, 0.945, 0.910, 0.680, 0.450], [0.985, 0.910, 0.650, 0.350, 0.180],
    [0.972, 0.680, 0.350, 0.180, 0.090], [0.991, 0.955, 0.938, 0.925, 0.918]
]
rec_data = [
    [0.951, 0.930, 0.880, 0.520, 0.250], [0.933, 0.680, 0.490, 0.290, 0.150],
    [0.958, 0.880, 0.780, 0.280, 0.110], [0.924, 0.820, 0.310, 0.080, 0.030],
    [0.984, 0.380, 0.080, 0.020, 0.010], [0.990, 0.885, 0.821, 0.795, 0.788]
]
f1_data = [
    [0.9427, 0.9043, 0.8647, 0.6040, 0.3547], [0.9384, 0.7595, 0.5800, 0.3809, 0.2210],
    [0.9615, 0.9113, 0.8402, 0.3967, 0.1768], [0.9535, 0.8624, 0.4198, 0.1302, 0.0514],
    [0.9780, 0.4875, 0.1302, 0.0360, 0.0180], [0.9915, 0.9187, 0.8756, 0.8551, 0.8481]
]

all_data = [acc_data, pre_data, rec_data, f1_data]

# 所有的文本都交回给宋体
titles_cn = ['准确率', '精确率', '召回率', 'F1分数']
y_labels_cn = ['准确率', '精确率', '召回率', 'F1分数']

markers = ['o', 's', '^', 'D', 'v', '*']
linestyles = ['--', '-.', ':', '--', '-.', '-']
colors = ['#8C8C8C', '#5A9BD5', '#70AD47', '#FFC000', '#ED7D31', '#C00000'] 

# ==================== 3. 绘图执行 ====================
fig, axs = plt.subplots(2, 2, figsize=(14.5, 9.5)) 
axs = axs.flatten() 

for i in range(4):
    ax = axs[i]
    for j in range(6):
        lw = 2.5 if j == 5 else 1.5
        ms = 10 if j == 5 else 6
        ax.plot(epsilons, all_data[i][j], marker=markers[j], linestyle=linestyles[j], 
                color=colors[j], linewidth=lw, markersize=ms, label=models[j], clip_on=False)
    
    # X轴和Y轴标签全部应用本地宋体
    ax.set_xlabel(r'对抗扰动半径 $\epsilon$', fontproperties=font_cn, fontsize=18)
    ax.set_ylabel(y_labels_cn[i], fontproperties=font_cn, fontsize=18)
    
    # ================= 核心拆分：仅针对 (a)(b)(c)(d) =================
    ax.set_title('') # 清空默认标题
    
    # 获取当前子图对应的字母，例如 (a), (b)...
    letter = f'({chr(97+i)})'
    
    # 字母绑定 TIMES.TTF，靠左排布；中文绑定 SIMSUN.TTC，靠右排布，完美拼接
    ax.text(0.48, -0.26, letter, transform=ax.transAxes, fontproperties=font_en, ha='right', va='center')
    ax.text(0.50, -0.26, titles_cn[i], transform=ax.transAxes, fontproperties=font_cn, ha='left', va='center')

    ax.set_xlim(0.0, 0.8)
    ax.set_xticks(epsilons)
    ax.set_ylim(0.0, 1.0) 
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
    
    ax.grid(axis='both', linestyle='--', alpha=0.6)
    
    # 强制刻度数字使用本地 TIMES.TTF
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_ticks)

# 布局调整
plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1, hspace=0.37, wspace=0.6)

# 强制图例使用本地 TIMES.TTF
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center', ncol=1, prop=font_legend, 
           frameon=True, edgecolor='black', facecolor='white', framealpha=1, bbox_to_anchor=(0.505, 0.5))

plt.savefig('抗扰动能力对比图.png', dpi=300, bbox_inches='tight')
print("绘制成功！纵坐标和 F1分数 已保持宋体，(a)(b)(c)(d) 已完美独立并呈现 Times New Roman。")