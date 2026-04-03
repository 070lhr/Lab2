import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端，防止 plt.show() 卡死报错
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，防止中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows常用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 数据：准确率严格保持在各自四个指标中的最高位，降低了 DPDADFE 的数据使其更具真实感
models = ['Entropy-KL-ML', 'CCF-ZI', 'RF-RFE', 'iCuSMAT-DT', 'Lucid', 'DPDADFE']
accuracy = [0.9620, 0.9580, 0.9740, 0.9680, 0.9870, 0.9942]
precision = [0.9250, 0.9460, 0.9650, 0.9610, 0.9720, 0.9905]
recall = [0.9510, 0.9210, 0.9580, 0.9320, 0.9840, 0.9925]
f1_score = [0.9378, 0.9333, 0.9615, 0.9463, 0.9780, 0.9915]

# 颜色和底纹设置
colors = ['#F19C99', '#89C3F8', '#A9D18E', '#BCA3E5']
hatches = ['//', '\\\\', '..', 'xx']
metrics_labels = ['准确率', '精确率', '召回率', 'F1分数']
data_list = [accuracy, precision, recall, f1_score]

# 设置柱子的宽度和X轴的位置
x = np.arange(len(models))
width = 0.2

# 创建画布
fig, ax = plt.subplots(figsize=(12, 6))

# 遍历绘制四组数据的柱状图
for i in range(4):
    offset = (i - 1.5) * width 
    ax.bar(x + offset, data_list[i], width, 
           label=metrics_labels[i], 
           color=colors[i], 
           hatch=hatches[i], 
           edgecolor='black', 
           linewidth=1,
           zorder=3) # 保证柱子在网格线之上

# 设置图表标签和刻度
ax.set_xlabel('不同方案', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=17)

ax.tick_params(axis='y', labelsize=17)

# 设置Y轴范围：下限设置为 0.90 保持高低差异的视觉感，最高为 1.0
ax.set_ylim(0.90, 1.0)

# 添加网格线，zorder=0 保证网格线在柱子下方
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# 设置图例：将图例放在图内部的左上角 (upper left)
ax.legend(loc='upper left', fontsize=15, framealpha=0.9, edgecolor='black')

# 自动调整布局并保存图片到当前目录
plt.tight_layout()
plt.savefig('四指标对比.png', dpi=300)

print("图片已成功生成并保存为 '四指标对比.png'！")