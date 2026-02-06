import pandas as pd
import numpy as np

# 输入/输出文件
INPUT_FILE = './flash_event_cleaned.csv'
OUTPUT_FILE = './flash_event_9dim_final.csv'

def compute_advanced_features(df):
    # 1. 确保按时间排序 (滑动窗口计算的前提)
    df = df.sort_values('timestamp')
    
    # === 第一组：速率维度 ===
    # f1: Rate (已有)
    # f2: Accel (已有，或者重新算 diff)
    df['Rate_Accel'] = df['Rate'].diff().fillna(0)
    # f3: 速率波动率 (5秒滑动窗口的标准差) -> 反映流量是否"死板"
    df['Rate_Vol'] = df['Rate'].rolling(window=5, min_periods=1).std().fillna(0)
    
    # === 第二组：熵维度 ===
    # f4: Entropy (已有)
    # f5: 熵的变化率
    df['Ent_Change'] = df['Entropy'].diff().fillna(0)
    # f6: 熵的移动平均 (5秒) -> 消除瞬间抖动，看长期分布
    df['Ent_MA'] = df['Entropy'].rolling(window=5, min_periods=1).mean().fillna(df['Entropy'])
    
    # === 第三组：载荷维度 ===
    # f7: Size_Std (已有)
    # f8: 大小标准差的变化
    df['SizeStd_Change'] = df['Size_Std'].diff().fillna(0)
    # f9: 大小标准差的均值 (5秒)
    df['SizeStd_MA'] = df['Size_Std'].rolling(window=5, min_periods=1).mean().fillna(df['Size_Std'])
    
    # 清理 NaN (第一行diff会产生NaN)
    df = df.fillna(0)
    
    # 只保留这 9 个特征 + Label
    cols = ['Rate', 'Rate_Accel', 'Rate_Vol', 
            'Entropy', 'Ent_Change', 'Ent_MA', 
            'Size_Std', 'SizeStd_Change', 'SizeStd_MA', 
            'Label']
    
    return df[cols]

if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    print(f"[*] 正在处理 Flash 数据: {len(df)} 条")
    
    df_new = compute_advanced_features(df)
    
    df_new.to_csv(OUTPUT_FILE, index=False)
    print(f"[*] Flash 9维特征已保存: {OUTPUT_FILE}")
    print(df_new.head())