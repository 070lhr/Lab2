import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import os

# ==================== 1. 字体配置 (精准模式) ====================
# 获取宋体文件路径
font_path = r"C:\Windows\Fonts\simsun.ttc"
if not os.path.exists(font_path):
    font_path = r"C:\Windows\Fonts\simhei.ttf" # 备选黑体

# 创建中文字体属性对象
font_cn = font_manager.FontProperties(fname=font_path)

# 全局默认字体设为 Times New Roman (负责所有的英文模型名、数字刻度)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False 

# ==================== 2. 数据准备 (保持不变) ====================
models = ['Entropy-KL-ML', 'CCF-ZI', 'RF-RFE', 'iCuSMAT-DT', 'Lucid', 'DPDADFE']
accuracy = [0.9620, 0.9580, 0.9740, 0.9680, 0.9870, 0.9942]
precision = [0.9250, 0.9460, 0.9650, 0.9610, 0.9720, 0.9905]
recall = [0.9510, 0.9210, 0.9580, 0.9320, 0.9840, 0.9925]
f1_score = [0.9378, 0.9333, 0.9615, 0.9463, 0.9780, 0.9915]

colors = ['#F19C99', '#89C3F8', '#A9D18E', '#BCA3E5']
hatches = ['//', '\\\\', '..', 'xx']
metrics_labels = ['准确率', '精确率', '召回率', 'F1分数']
data_list = [accuracy, precision, recall, f1_score]

x = np.arange(len(models))
width = 0.18 # 稍微调窄一点点，防止柱子太挤

# ==================== 3. 绘图执行 ====================
fig, ax = plt.subplots(figsize=(12, 6))

for i in range(4):
    offset = (i - 1.5) * width 
    ax.bar(x + offset, data_list[i], width, 
            label=metrics_labels[i], 
            color=colors[i], 
            hatch=hatches[i], 
            edgecolor='black', 
            linewidth=1,
            zorder=3)

# ==================== 4. 细节修饰与字体注入 ====================
# 【字体注入】设置 X 轴中文标题为宋体
ax.set_xlabel('不同方案', fontproperties=font_cn, fontsize=17)

# 设置 X 轴刻度：模型名全是英文，会自动使用 Times New Roman
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=17) # 字号微调以适应长度

# Y 轴数字刻度自动应用 Times New Roman
ax.tick_params(axis='y', labelsize=17)
ax.set_ylim(0.90, 1.0)

ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# 【字体注入】图例：包含中文指标名，需要显式指定宋体
# 复制一份字体属性并设置字号
font_cn_legend = font_cn.copy()
font_cn_legend.set_size(15)
ax.legend(loc='upper left', prop=font_cn_legend, framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.savefig('四指标对比.png', dpi=300, bbox_inches='tight')

print("图片已成功生成，已应用宋体(中)与Times New Roman(英/数)！")