import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
OUTPUT_IMG = '对抗攻击四指标退化组图.png'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文标签
plt.rcParams['axes.unicode_minus'] = False    # 负号
# ===========================================

def plot_4x4_real_robustness_grid():
    # 1. 设定横坐标：对抗扰动强度 (Epsilon)
    epsilons = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    # 2. 载入真实实验数据
    
    # --- (a) Accuracy 数据 ---
    acc_dt  = [90.15, 60.38, 56.96, 55.04, 54.26]
    acc_dnn = [94.62, 63.51, 50.14, 49.99, 49.99]
    acc_tcn = [97.45, 60.90, 49.98, 49.98, 49.98]
    acc_dpg = [100.00,90.50, 86.75, 86.02, 86.02]

    # --- (b) Precision 数据 ---
    pre_dt  = [89.85, 89.85, 89.85, 89.85, 89.85]
    pre_dnn = [94.18, 99.88, 90.32, 33.33, 14.29]
    pre_tcn = [97.20, 99.82, 0.00, 0.00, 0.00]
    pre_dpg = [100.00, 100.00, 100.00, 100.00, 100.00]

    # --- (c) Recall 数据 ---
    rec_dt  = [90.50, 20.77, 13.92, 10.09, 8.53]
    rec_dnn = [95.10, 27.04, 0.31, 0.02, 0.01]
    rec_tcn = [97.85, 21.84, 0.00, 0.00, 0.00]
    rec_dpg = [100.00, 81.00, 73.50, 72.05, 72.05]

    # --- (d) F1-Score 数据 ---
    f1_dt  = [90.17, 34.39, 24.43, 18.32, 15.72]
    f1_dnn = [94.64, 42.56, 0.62, 0.03, 0.01]
    f1_tcn = [97.52, 35.84, 0.00, 0.00, 0.00]
    f1_dpg = [100.00, 89.50, 84.73, 83.75, 83.75]

    # 3. 创建 2x2 的子图画布 (结合之前柱状图的经验，稍微压低高度让版面更致密)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5))
    
    # 统一线条样式
    styles = {
        'DT':  {'color': '#808080', 'ls': '--', 'marker': 'o', 'label': 'DT-Model'},
        'DNN': {'color': '#4A90E2', 'ls': '-.', 'marker': 's', 'label': 'DNN'},
        'TCN': {'color': '#F5A623', 'ls': ':',  'marker': '^', 'label': 'TCN'},
        'DPG': {'color': '#D0021B', 'ls': '-',  'marker': '*', 'label': 'DPG-Net', 'lw': 3.5, 'ms': 12}
    }

    def plot_subplot(ax, data_dt, data_dnn, data_tcn, data_dpg, ylabel, sub_id):
        ax.plot(epsilons, data_dt,  linewidth=2, markersize=7, **styles['DT'])
        ax.plot(epsilons, data_dnn, linewidth=2, markersize=7, **styles['DNN'])
        ax.plot(epsilons, data_tcn, linewidth=2, markersize=8, **styles['TCN'])
        ax.plot(epsilons, data_dpg, linewidth=styles['DPG']['lw'], markersize=styles['DPG']['ms'], 
                color=styles['DPG']['color'], linestyle=styles['DPG']['ls'], marker=styles['DPG']['marker'], label=styles['DPG']['label'])
        
        ax.set_ylim(-2, 105) 
        ax.set_xlim(-0.02, 0.82)
        ax.set_xticks(epsilons)
        ax.set_xticklabels([f'{e:.1f}' for e in epsilons], fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        
        # 【修改 1】：将原先的 Epsilon 替换为数学符号 $\epsilon$ (使用 rf 字符串)
        ax.set_xlabel(rf'$\epsilon$'+'\n'+f'{sub_id}', fontsize=16, fontweight='bold', labelpad=2)

    # 4. 绘图
    plot_subplot(axs[0, 0], acc_dt, acc_dnn, acc_tcn, acc_dpg, '准确率(%)', '(a)')
    plot_subplot(axs[0, 1], pre_dt, pre_dnn, pre_tcn, pre_dpg, '精确率(%)', '(b)')
    plot_subplot(axs[1, 0], rec_dt, rec_dnn, rec_tcn, rec_dpg, '召回率(%)', '(c)')
    plot_subplot(axs[1, 1], f1_dt, f1_dnn, f1_tcn, f1_dpg, 'F1分数(%)', '(d)')

    # 5. 图例定位
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.98), 
               ncol=4, fontsize=14, fancybox=True, shadow=True, framealpha=0.95)

    # 【修改 2】：紧凑化布局
    plt.tight_layout(h_pad=0.8, w_pad=2.0, rect=[0, 0, 1, 0.96])
    
    # 6. 保存图片
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"\n[+] 完美！数学符号版退化组图已生成并保存为: {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_4x4_real_robustness_grid()