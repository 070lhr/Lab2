import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
OUTPUT_IMG = '消融实验2x2组图.png'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文标签
plt.rcParams['axes.unicode_minus'] = False    # 负号
# ===========================================

accuracy = [0.9620, 0.9580, 0.9740, 0.9680, 0.9870, 0.9942]
precision = [0.9250, 0.9460, 0.9650, 0.9610, 0.9720, 0.9905]
recall = [0.9510, 0.9210, 0.9580, 0.9320, 0.9840, 0.9925]
f1_score = [0.9378, 0.9333, 0.9615, 0.9463, 0.9780, 0.9915]
def plot_4x4_real_ablation_grid():
    labels = ['Dist-Only', 'Dyn-Only', 'DPDADFE']
    x = np.arange(len(labels))
    width = 0.35  

    # 载入真实实验数据
    acc_clean = [99.85, 99.99, 0.9942]
    acc_adv   = [73.75, 49.99, 0.888]
    
    pre_clean = [99.75, 99.97, 0.9905]
    pre_adv   = [99.48,  0.00, 0.918]
    
    rec_clean = [99.95, 100.00, 0.9925]
    rec_adv   = [47.75,   0.00,  0.788]
    
    f1_clean  = [99.85, 99.99, 0.9915]
    f1_adv    = [64.53,  0.00,  0.8481]

    # 将高度从 9 压缩到 7.5，物理上挤压多余的白边
    fig, axs = plt.subplots(2, 2, figsize=(14, 7.5))

    def plot_bar_subplot(ax, data_clean, data_adv, ylabel, sub_id):
        # 【修改点】使用 r 字符串和 LaTeX 语法渲染 \epsilon
        rects1 = ax.bar(x - width/2, data_clean, width, label=r'正常环境 ($\epsilon=0.0$)', color='#4A90E2', edgecolor='black', alpha=0.9)
        rects2 = ax.bar(x + width/2, data_adv, width, label=r'对抗攻击 ($\epsilon=0.8$)', color='#D0021B', edgecolor='black', alpha=0.9)
        
        ax.set_ylim(0, 119) # 留出顶部写数值的空间
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        
        # 将 labelpad 缩小到 2，让 (a)(b) 紧贴上面的文字
        ax.set_xlabel(sub_id, fontsize=14, fontweight='bold', labelpad=2)

        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
        autolabel(rects1)
        autolabel(rects2)

    # 将具体的指标名称传入作为 ylabel
    plot_bar_subplot(axs[0, 0], acc_clean, acc_adv, '准确率(%)', '(a)')
    plot_bar_subplot(axs[0, 1], pre_clean, pre_adv, '精确率(%)', '(b)')
    plot_bar_subplot(axs[1, 0], rec_clean, rec_adv, '召回率(%)', '(c)')
    plot_bar_subplot(axs[1, 1], f1_clean, f1_adv, 'F1分数(%)', '(d)')

    # 统一图例 (微调了 bbox_to_anchor 配合更扁的画板)
    handles, labels_legend = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=2, fontsize=14, fancybox=True, shadow=True)

    # 将 h_pad (垂直填充距) 从 2.0 暴降到 0.5，彻底缝合上下两排
    plt.tight_layout(h_pad=0.5, w_pad=2.0, rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"\n[+] 包含标准数学符号的矩阵图已生成！请查看: {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_4x4_real_ablation_grid()