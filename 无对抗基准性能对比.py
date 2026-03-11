import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
OUTPUT_IMG = '基准性能对比图.png'
# 设置东亚文字字体，确保中文正常显示 (如果是在本地Windows运行)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# 如果是Linux服务器，请确保安装了黑体，或者使用您之前的 MSYH.TTC 方法
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# ===========================================

def plot_performance_bar():
    # 1. 准备核心数据 (来自 4.5.2 节的终极定稿)
    metrics = ['准确率', '精确率', '召回率', 'F1分数']  
    
    # 四个模型的数据
    dt_data  = [90.15, 89.85, 90.50, 90.17]
    dnn_data = [94.62, 94.18, 95.10, 94.64]
    tcn_data = [97.45, 97.20, 97.85, 97.52]
    dpg_data = [100.00, 100.00, 100.00, 100.00]

    # 2. 设置柱状图的位置与宽度
    x = np.arange(len(metrics))  
    width = 0.2  # 每个柱子的宽度
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. 绘制分组柱状图 (【关键修改】全部去除 hatch 参数，改为纯色填充)
    # DT: 纯灰色
    rects1 = ax.bar(x - 1.5*width, dt_data, width, label='DT-Model', 
                    color='#B0B0B0', edgecolor='black')
    # DNN: 纯学术蓝
    rects2 = ax.bar(x - 0.5*width, dnn_data, width, label='DNN', 
                    color='#4A90E2', edgecolor='black')
    # TCN: 纯活力橙
    rects3 = ax.bar(x + 0.5*width, tcn_data, width, label='TCN', 
                    color='#F5A623', edgecolor='black')
    # DPG-Net: 纯霸气深红
    rects4 = ax.bar(x + 1.5*width, dpg_data, width, label='DPG-Net', 
                    color='#D0021B', edgecolor='black')

    # 4. 图表装饰与标签设置
    # 将 Y 轴范围设为 0 到 105，清爽大气
    ax.set_ylim(0, 10)
    ax.set_yticks(np.arange(0, 101, 20))
    
    ax.set_ylabel('百分比 (%)', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14, fontweight='bold')
    
    # 图例放在图外，不遮挡柱子
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
              ncol=4, fancybox=True, shadow=True, fontsize=12)

    # 添加 Y 轴辅助网格线，让对比更清晰
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 自动紧凑布局
    plt.tight_layout()
    
    # 5. 保存图片 (只保存，不显示，完美兼容服务器无头环境)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"\n[+] 完美！纯色无纹理的清爽版柱状图已生成并保存为: {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_performance_bar()