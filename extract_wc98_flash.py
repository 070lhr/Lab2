import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import Counter
import os

# ================= 配置区域 =================
# 输入：WC98 Day 73 的原始 CSV 路径
# (确保这个 CSV 里有 timestamp, clientID, size 这三列)
INPUT_WC_CSV = '/data/exp/hrliu/1998WC/WorldCupCSV/wc_day73_1.csv' 

# 输出：处理好并筛选过的特征文件
OUTPUT_CSV = './wc98_flash_event_features.csv'

# 标签：Flash Event 标记为 0
LABEL = 0

# 筛选策略：只保留速率排名前 N 的秒数 (例如 30000 秒)
# 这样可以保证你拿到的都是"真·Flash Event"，且数量与 DDoS 平衡
TOP_N_SECONDS = 30000 
# ===========================================

def calculate_entropy(ids):
    """ 计算 clientID 的香农熵 """
    if len(ids) == 0: return 0
    counts = np.array(list(Counter(ids).values()))
    probs = counts / len(ids)
    return entropy(probs, base=2)

def process_wc98(input_path, output_path):
    print(f"[*] 正在读取 WC98 数据: {input_path} ...")
    
    if not os.path.exists(input_path):
        print(f"[!] 错误: 找不到文件 {input_path}")
        return

    # 1. 读取数据 (使用 chunksize 防止内存爆炸，因为 WC98 文件可能很大)
    # 假设 CSV 没有表头，视情况修改 header=None 或 header=0
    # WC98 标准格式通常是: timestamp, clientID, objectID, size, method, status, type, server
    # 这里我们假设您已经整理成了带表头的 CSV，或者您根据实际情况修改 names
    try:
        # 如果您的 CSV 有表头，去掉 names 参数；如果没有，请加上 names
        # 这里假设您的 CSV 已经有了标准的表头: timestamp, clientID, size
        df_chunks = pd.read_csv(input_path, usecols=['timestamp', 'clientID', 'size'], chunksize=1000000)
    except ValueError:
        print("[!] 列名匹配失败，尝试使用默认列名读取...")
        # 备用方案：如果是原始格式，可能没有表头，需要手动指定
        df_chunks = pd.read_csv(input_path, header=None, names=['timestamp', 'clientID', 'objectID', 'size', 'method', 'status', 'type', 'server'], usecols=['timestamp', 'clientID', 'size'], chunksize=1000000)

    # 2. 分块聚合 (Map-Reduce 思路)
    print("[*] 正在分块聚合数据 (按秒统计)...")
    aggregated_data = []

    for chunk in df_chunks:
        # 按 timestamp 分组计算
        # count -> Rate
        # std(size) -> Size_Std
        # list(clientID) -> 稍后算 Entropy
        
        # 这里的技巧：为了算 Entropy，我们需要保留这一秒所有的 clientID
        # 但这样内存会爆。所以我们先只聚合 Rate 和 Size_Std
        # Entropy 我们用 "Unique Client Count" 近似，或者如果不算 Entropy 也可以
        # 为了严谨，建议针对 chunk 计算，但跨 chunk 的 timestamp 会比较麻烦
        
        # === 简化版高效方案 ===
        # 直接把整个 chunk 按 timestamp 分组
        grouped = chunk.groupby('timestamp')
        
        for ts, group in grouped:
            aggregated_data.append({
                'timestamp': ts,
                'Rate': len(group),
                'Size_Std': group['size'].std(),
                'Client_IDs': group['clientID'].tolist() # 先存着，最后合并算熵? 不，内存不够。
                # 修正：直接在这里算熵。虽然同一个 timestamp 可能跨 chunk，但概率较小
                # 或者我们就假设 chunk 足够大覆盖了完整的几秒
            })
            
    # 将列表转为 DataFrame
    print("[*] 正在合并分块结果...")
    df_temp = pd.DataFrame(aggregated_data)
    
    # 再次聚合：因为同一个 timestamp 可能出现在两个 chunk 的交界处
    # 我们需要把相同 timestamp 的行合并
    # 这是一个比较重的操作，为了计算 Entropy 准确，我们需要合并 clientID 列表
    # 但为了性能，我们可以简化：假设 chunk 没切断同一秒 (大部分情况是这样的)
    
    # 如果您追求极致严谨，需要把 Client_IDs extend 起来再算。
    # 这里我们采用直接计算法 (假设 chunk 切分对秒级统计影响可忽略)
    
    features = []
    print("[*] 正在计算最终特征 (Rate, Std, Entropy)...")
    
    # 这里我们需要重新整理数据
    # 为了简单起见，我们重新遍历 df_temp
    # 注意：上面的 aggregated_data 里存了 client_ids 可能会撑爆内存
    # 更好的方法是在上面的循环里直接算好 Entropy。
    
    # === 修正后的流式处理逻辑 (内存安全) ===
    # 让我们重新写一个更简单的逻辑：直接读取并计算
    pass 

# ==============================================================================
# 上面的逻辑有点复杂，容易内存溢出。
# 让我们换一个更直接、更稳健的 pandas 逻辑 (假设内存够载入 Day73 的几列)
# ==============================================================================

def process_wc98_simple(input_path, output_path):
    print(f"[*] 正在读取 WC98 数据 (这可能需要一点时间)...")
    
    # 读取所有数据 (只读需要的 3 列，通常 Day73 也就几百兆，内存够用)
    try:
        df = pd.read_csv(input_path, usecols=['timestamp', 'clientID', 'size'])
    except:
        # 适配无表头情况
        df = pd.read_csv(input_path, header=None, sep=' ', names=['timestamp', 'clientID', 'objectID', 'size', 'method', 'status', 'type', 'server'], usecols=['timestamp', 'clientID', 'size'])

    print(f"[*] 数据加载完成，共 {len(df)} 条请求。开始聚合...")

    # 1. 按 timestamp 分组
    grouped = df.groupby('timestamp')

    # 2. 计算基础特征
    #    Rate = count()
    #    Size_Std = std(size)
    #    Entropy = apply(calculate_entropy)
    
    # 为了速度，我们手动迭代，或者用 agg
    # 自定义聚合函数
    def agg_func(x):
        d = {}
        d['Rate'] = x['clientID'].count()
        d['Size_Std'] = x['size'].std()
        d['Entropy'] = calculate_entropy(x['clientID'])
        return pd.Series(d)

    # 这里的 apply 可能会慢，显示个进度条
    from tqdm import tqdm
    tqdm.pandas(desc="计算特征中")
    
    df_features = grouped.progress_apply(agg_func).reset_index()
    
    # 填补 Size_Std 的 NaN (如果这一秒只有一个请求，std 是 NaN)
    df_features['Size_Std'] = df_features['Size_Std'].fillna(0)

    # 3. 计算第 4 个特征：Accel (加速度)
    print("[*] 正在计算加速度 (Accel)...")
    df_features = df_features.sort_values('timestamp')
    df_features['Accel'] = df_features['Rate'].diff().fillna(0)

    # 4. 关键步骤：筛选 Flash Event (提取峰值)
    print(f"[*] 正在筛选 Top {TOP_N_SECONDS} 高并发样本...")
    
    # 按 Rate 从大到小排序
    df_sorted = df_features.sort_values(by='Rate', ascending=False)
    
    # 取前 N 个
    df_flash = df_sorted.head(TOP_N_SECONDS).copy()
    
    # 打上标签
    df_flash['Label'] = LABEL
    
    # 重新按时间排序 (为了保持时序性，方便后面切分序列)
    df_flash = df_flash.sort_values('timestamp')

    # 5. 保存
    # 调整列顺序
    cols = ['timestamp', 'Rate', 'Size_Std', 'Entropy', 'Accel', 'Label']
    df_flash = df_flash[cols]
    
    df_flash.to_csv(output_path, index=False)
    print(f"[*] 处理完成！")
    print(f"    - 原始秒数: {len(df_features)}")
    print(f"    - 筛选后秒数: {len(df_flash)}")
    print(f"    - 输出文件: {os.path.abspath(output_path)}")
    print("\n数据预览:")
    print(df_flash.head())

if __name__ == "__main__":
    # 需要安装: pip install pandas numpy scipy tqdm
    process_wc98_simple(INPUT_WC_CSV, OUTPUT_CSV)