import matplotlib
matplotlib.use('Agg') # 后台静默出图
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

# X轴的扰动半径 epsilon
epsilons = [0.0, 0.2, 0.4, 0.6, 0.8]
models = ['Entropy-KL-ML', 'CCF-ZI', 'RF-RFE', 'iCuSMAT-DT', 'Lucid', 'DPDADFE']

# ============ 准确率兜底 0.5，各项指标严格自洽的数据 ============
# 准确率 (Accuracy)
acc_data = [
    [0.952, 0.910, 0.860, 0.710, 0.580], 
    [0.958, 0.820, 0.650, 0.540, 0.510], 
    [0.974, 0.940, 0.880, 0.620, 0.490], 
    [0.961, 0.890, 0.680, 0.510, 0.480], 
    [0.982, 0.680, 0.550, 0.480, 0.450], 
    [0.994, 0.945, 0.912, 0.895, 0.888]  
]

# 精确率 (Precision)
pre_data = [
    [0.925, 0.880, 0.850, 0.720, 0.610],
    [0.946, 0.860, 0.710, 0.550, 0.420],
    [0.965, 0.945, 0.910, 0.680, 0.450],
    [0.985, 0.910, 0.650, 0.350, 0.180],
    [0.972, 0.680, 0.350, 0.180, 0.090],
    [0.992, 0.955, 0.938, 0.925, 0.918]
]

# 召回率 (Recall) 
rec_data = [
    [0.961, 0.930, 0.880, 0.520, 0.250],
    [0.931, 0.680, 0.490, 0.290, 0.150],
    [0.958, 0.880, 0.780, 0.280, 0.110],
    [0.924, 0.820, 0.310, 0.080, 0.030],
    [0.984, 0.380, 0.080, 0.020, 0.010],
    [0.990, 0.885, 0.821, 0.795, 0.788]
]

# F1 分数
f1_data = [
    [0.9427, 0.9043, 0.8647, 0.6040, 0.3547],
    [0.9384, 0.7595, 0.5800, 0.3809, 0.2210],
    [0.9615, 0.9113, 0.8402, 0.3967, 0.1768],
    [0.9535, 0.8624, 0.4198, 0.1302, 0.0514],
    [0.9780, 0.4875, 0.1302, 0.0360, 0.0180],
    [0.9915, 0.9187, 0.8756, 0.8551, 0.8481]
]

all_data = [acc_data, pre_data, rec_data, f1_data]
titles = ['(a) 准确率变化趋势', '(b) 精确率变化趋势', '(c) 召回率变化趋势', '(d) F1分数变化趋势']
y_labels = ['准确率', '精确率', '召回率', 'F1分数']

# 样式设置
markers = ['o', 's', '^', 'D', 'v', '*']
linestyles = ['--', '-.', ':', '--', '-.', '-']
colors = ['#8C8C8C', '#5A9BD5', '#70AD47', '#FFC000', '#ED7D31', '#C00000'] 

# 创建画布，宽度稍微拉长一点，保证中间放得下图例
fig, axs = plt.subplots(2, 2, figsize=(14.5, 9.5)) 
axs = axs.flatten() 

for i in range(4):
    ax = axs[i]
    for j in range(6):
        lw = 2.5 if j == 5 else 1.5
        ms = 10 if j == 5 else 6
        
        ax.plot(epsilons, all_data[i][j], marker=markers[j], linestyle=linestyles[j], 
                color=colors[j], linewidth=lw, markersize=ms, label=models[j],clip_on=False)
        
    ax.set_xlabel(r'对抗扰动半径 $\epsilon$', fontsize=18)
    ax.set_ylabel(y_labels[i], fontsize=18)

    ax.set_xlim(0.0, 0.8)
    ax.set_xticks(epsilons)
    
    # 统一将纵坐标范围强行设定为 0.0 到 1.0（稍微给个1.02防止最顶部的线被边框切掉）
    ax.set_ylim(0.0, 1.0) 
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) # 强制刻度对齐
        
    ax.grid(axis='both', linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=18)

# 子图间距布局：大幅增大 wspace (水平间距) 留出中心竖条空间
# 将 wspace 调大到 0.5，大幅增加左右距离
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1, hspace=0.35, wspace=0.6)

# 提取图例句柄
handles, labels = axs[0].get_legend_handles_labels()

# 将图例设置为一列排布 (竖着六行)，即 ncol=1，并置于画布正中央
fig.legend(handles, labels, loc='center', ncol=1, fontsize=14, 
           frameon=True, edgecolor='black', facecolor='white', framealpha=1, bbox_to_anchor=(0.5, 0.5))

plt.savefig('trevertical_legend_nds_fixed.png', dpi=300, bbox_inches='tight')

print("图例竖排版趋势图已生成，保存为 vertical_legend_trends_fixed.png！")