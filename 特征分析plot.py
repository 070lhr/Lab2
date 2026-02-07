import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# 读取数据

df_flash = pd.read_csv('./flash_event_9dim_final.csv')

df_ddos = pd.read_csv('./ciciot_ddos_9dim_final.csv')



# 选择关键特征

features_to_check = ['Size_Std', 'Entropy', 'Rate_Vol']



plt.figure(figsize=(15, 5))



for i, col in enumerate(features_to_check):

    plt.subplot(1, 3, i+1)

    # 画重叠直方图

    sns.histplot(df_flash[col], color='blue', label='Flash', kde=True, stat="density", common_norm=False)

    sns.histplot(df_ddos[col], color='red', label='DDoS', kde=True, stat="density", common_norm=False)

    plt.title(f'Feature Distribution: {col}')

    plt.legend()



plt.tight_layout()

plt.savefig('debug_distribution.png')

print("分布图已保存为 debug_distribution.png，请查看是否有特征完全不重叠。")