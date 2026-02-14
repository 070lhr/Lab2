import pandas as pd
import os

# ================= 配置区域 =================
# 1. 输入文件 (您想过滤的那个文件)
# 如果是那个 49万条的大文件，请把这里改成 'flash_event_9dim_ALL_PEAKS.csv'
INPUT_FILE = './flash_event_9dim_full.csv'

# 2. 输出文件 (过滤后的新文件)
OUTPUT_FILE = './flash_event_9dim_filtered_1200.csv'

# 3. 过滤阈值 (只保留 Rate >= 1200 的行)
THRESHOLD = 1200
# ===========================================

def main():
    print(f"[*] 正在读取文件: {INPUT_FILE} ...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"[!] 错误: 找不到文件 {INPUT_FILE}")
        return

    # 读取数据
    df = pd.read_csv(INPUT_FILE)
    total_count = len(df)
    
    print(f"[*] 原始样本数: {total_count}")
    
    # === 核心过滤逻辑 ===
    # 保留 Rate >= THRESHOLD 的数据
    df_filtered = df[df['Rate'] >= THRESHOLD].copy()
    filtered_count = len(df_filtered)
    
    # 计算被移除的数量
    removed_count = total_count - filtered_count
    retention_rate = (filtered_count / total_count) * 100

    print("-" * 40)
    print(f"[*] 过滤条件: Rate >= {THRESHOLD}")
    print(f"[*] 保留样本数: {filtered_count}")
    print(f"[*] 移除样本数: {removed_count}")
    print(f"[*] 保留比例:   {retention_rate:.2f}%")
    print("-" * 40)

    if filtered_count == 0:
        print("[!] 警告: 过滤后没有剩余数据！请检查阈值是否过高。")
        return

    # 保存文件
    df_filtered.to_csv(OUTPUT_FILE, index=False)
    print(f"[*] 成功！新文件已保存至: {OUTPUT_FILE}")
    print(f"   (包含 {filtered_count} 条高质量 Flash 高峰样本)")

if __name__ == "__main__":
    main()