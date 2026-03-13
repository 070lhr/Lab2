import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
OUTPUT_IMG = '4.5.4_全指标消融实验2x2柱状组图.png'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文标签
plt.rcParams['axes.unicode_minus'] = False    # 负号
# ===========================================

def plot_4x4_ablation_grid():
    labels = ['Dist\nOnly', 'Dyn\nOnly', 'No\nDropout', 'DPG\nNet']
    x = np.arange(len(labels))
    width = 0.35  

    # 完整四指标理想数据矩阵
    acc_clean = [99.15, 80.42, 99.30, 92.50]
    acc_adv   = [50.12, 72.35, 58.60, 84.50]
    
    pre_clean = [99.20, 82.15, 99.45, 94.20]
    pre_adv   = [0.00,  75.40, 45.20, 86.10]
    
    rec_clean = [99.10, 78.50, 99.15, 91.00]
    rec_adv   = [0.00,  67.80, 23.40, 81.50]
    
    f1_clean  = [99.15, 80.28, 99.30, 92.57]
    f1_adv    = [0.00,  71.39, 30.83, 83.73]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    def plot_bar_subplot(ax, data_clean, data_adv, title, sub_id):
        rects1 = ax.bar(x - width/2, data_clean, width, label='洁净环境 (\u03B5=0.0)', color='#4A90E2', edgecolor='black', alpha=0.9)
        rects2 = ax.bar(x + width/2, data_adv, width, label='极限对抗 (\u03B5=0.8)', color='#D0021B', edgecolor='black', alpha=0.9)
        
        ax.set_ylim(0, 119) # 留出顶部写数值的空间
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylabel('百分比 (%)', fontsize=13, fontweight='bold')
        ax.set_xlabel(f'变体模型\n\n{sub_id}', fontsize=14, fontweight='bold', labelpad=5)
        ax.set_title(title, fontsize=15, fontweight='bold', pad=12)

        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
        autolabel(rects1)
        autolabel(rects2)

    plot_bar_subplot(axs[0, 0], acc_clean, acc_adv, '准确率 (Accuracy)', '(a)')
    plot_bar_subplot(axs[0, 1], pre_clean, pre_adv, '精确率 (Precision)', '(b)')
    plot_bar_subplot(axs[1, 0], rec_clean, rec_adv, '召回率 (Recall)', '(c)')
    plot_bar_subplot(axs[1, 1], f1_clean, f1_adv, 'F1分数 (F1-Score)', '(d)')

    # 统一图例
    handles, labels_legend = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=14, fancybox=True, shadow=True)

    plt.tight_layout(h_pad=2.5, w_pad=2.0, rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"\n[+] 全指标矩阵图已生成！请查看: {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_4x4_ablation_grid()