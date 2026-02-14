#!/usr/bin/env python3
import pandas as pd
import sys
import os

# ================= 配置区域 =================
# 这里填您那个 49万条的全量 Flash 文件路径
INPUT_FILE = './flash_event_9dim_full.csv'
# ===========================================

def main():
    print(f"[*] 正在读取文件: {INPUT_FILE} ...")
    if not os.path.exists(INPUT_FILE):
        print(f"[!] 错误: 文件不存在。请检查路径。")
        return

    try:
        # 只读取 Rate 列，速度飞快
        df = pd.read_csv(INPUT_FILE, usecols=['Rate'])
    except Exception as e:
        print(f"[!] 读取失败: {e}")
        return

    total_count = len(df)
    print(f"[*] 读取成功！总样本数: {total_count}")
    print("="*60)

    # 1. 自动计算分位点 (给您一个全局概览)
    print("【数据分布概览】")
    percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    desc = df['Rate'].describe(percentiles=percentiles)
    print(f"Min  : {desc['min']:.0f}")
    print(f"50%  : {desc['50%']:.0f} (中位数)")
    print(f"75%  : {desc['75%']:.0f}")
    print(f"90%  : {desc['90%']:.0f}")
    print(f"99%  : {desc['99%']:.0f}")
    print(f"Max  : {desc['max']:.0f}")
    print("-" * 60)

    # 2. 自动推荐阈值表
    # 帮您算好几个档位，看看哪个能凑够 10万~15万
    print("【推荐阈值表 (目标: 留下 10w~20w 样本)】")
    print(f"{'阈值 (Rate >)':<15} | {'保留样本数':<15} | {'保留比例':<10}")
    print("-" * 45)
    
    # 设定几个探测点，您可以根据上面的 max 自己调整
    check_points = [500, 800, 1000, 1200, 1500, 2000, 3000, 5000, 10000]
    
    for thresh in check_points:
        count = (df['Rate'] > thresh).sum()
        ratio = (count / total_count) * 100
        print(f"{thresh:<15} | {count:<15} | {ratio:.1f}%")
        
    print("="*60)

    # 3. 交互式查询
    print("输入任意数值查询 (输入 q 退出):")
    while True:
        user_input = input(">> 请输入 Rate 阈值: ").strip()
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        
        if not user_input.isdigit():
            print("[!] 请输入整数。")
            continue
            
        threshold = int(user_input)
        count = (df['Rate'] > threshold).sum()
        print(f"[*] Rate > {threshold} 的样本数: {count}")
        
        # 给个评价
        if 30000 <= count <= 150000:
            print("   ✅ 这个数量很棒！适合用来平衡您的 3万条 DDoS 数据 (1:1 ~ 5:1)。")
        elif count < 10000:
            print("   ⚠️ 太少了，可能会导致过拟合。")
        elif count > 200000:
            print("   ⚠️ 还是有点多，DDoS 数据可能会被淹没。")

if __name__ == "__main__":
    main()