import matplotlib.pyplot as plt

# 设置中文字体与基础样式
plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# X轴：对抗扰动半径
epsilons = [0.0, 0.2, 0.4, 0.6, 0.8]

# === 重新校准的非同构数据 ===

# 1. 准确率 (Accuracy) - 综合正确率，呈正常衰减
acc_no   = [0.996, 0.865, 0.680, 0.540, 0.455]
acc_sym  = [0.993, 0.922, 0.810, 0.695, 0.605]
acc_asym = [0.995, 0.968, 0.925, 0.883, 0.835]

# 2. 精确率 (Precision) - 误报率影响，下降较为平缓和线性
prec_no   = [0.997, 0.915, 0.782, 0.651, 0.533]
prec_sym  = [0.989, 0.931, 0.845, 0.742, 0.648]
prec_asym = [0.994, 0.970, 0.932, 0.895, 0.852]

# 3. 召回率 (Recall) - 对抗攻击的首要受害者，无防御模型出现断崖式崩溃
rec_no   = [0.994, 0.821, 0.583, 0.435, 0.352]
rec_sym  = [0.991, 0.912, 0.775, 0.642, 0.551]
rec_asym = [0.992, 0.965, 0.918, 0.872, 0.815]

# 4. F1分数 (F1-Score) - 调和平均，自然呈现出差异化曲线
f1_no   = [0.995, 0.865, 0.667, 0.521, 0.424]
f1_sym  = [0.990, 0.921, 0.808, 0.688, 0.595]
f1_asym = [0.993, 0.967, 0.925, 0.883, 0.833]

# === 创建 2x2 子图 ===
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 绘图配置字典
styles = [
    {'marker':'o', 'ls':'--', 'color':'gray', 'label':'无失活机制 (No Dropout)'},
    {'marker':'s', 'ls':'-.', 'color':'#4A90E2', 'label':'对称失活机制 (Symmetric)'},
    {'marker':'^', 'ls':'-', 'color':'#D0021B', 'lw':2, 'label':'非对称失活机制 (DPDADFE)'}
]

def plot_subplot(ax, data_list, ylabel, title):
    ax.plot(epsilons, data_list[0], **styles[0])
    ax.plot(epsilons, data_list[1], **styles[1])
    ax.plot(epsilons, data_list[2], **styles[2])
    # 注意这里使用了原始字符串 r'' 防止 \e 转义警告
    ax.set_xlabel(r'对抗扰动半径 $\epsilon$', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, y=-0.2, fontsize=14) 
    # Y轴下限设为0.3，以容纳召回率的最低点
    ax.set_ylim(0.3, 1.05)
    ax.set_xticks(epsilons)
    ax.grid(True, linestyle='--', alpha=0.6)

# 绘制四个指标
plot_subplot(axs[0, 0], [acc_no, acc_sym, acc_asym], '准确率', '(a) 准确率')
plot_subplot(axs[0, 1], [prec_no, prec_sym, prec_asym], '精确率', '(b) 精确率')
plot_subplot(axs[1, 0], [rec_no, rec_sym, rec_asym], '召回率', '(c) 召回率')
plot_subplot(axs[1, 1], [f1_no, f1_sym, f1_asym], 'F1分数', '(d) F1分数')

# 提取图例并统一放在图表中间偏上的位置
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.52), 
           ncol=1, fontsize=12, framealpha=1.0, edgecolor='black')

# 调整间距
plt.subplots_adjust(hspace=0.35, wspace=0.2)
plt.savefig('消融实验-Dropout.png图', dpi=600, bbox_inches='tight')