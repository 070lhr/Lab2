import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# ================= 配置区域 =================
# 1. 模型与数据路径
MODEL_PATH = './dpg_net_model.pth'
SCALER_PATH = './dpg_scaler.pkl'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'  # 您的 DDoS 数据集

# 2. 输出路径
ADV_OUTPUT_CSV = './adversarial_ddos_samples.csv'

# 3. 攻击配置
# 暴力搜索的 Epsilon 列表：从微扰(0.1)到强力摧毁(5.0)
EPSILONS_TO_SEARCH = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

# 4. 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

# --- 1. 模型定义 (必须与训练时完全一致) ---
class DPG_Net(nn.Module):
    def __init__(self):
        super(DPG_Net, self).__init__()
        
        # 动力学流
        self.dynamics_branch = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 分布流
        self.dist_branch = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # === 融合层 (修复点) ===
        # 必须保留 Dropout 层以匹配权重文件的结构
        self.fusion_layer = nn.Sequential(
            nn.Linear(16 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # <--- 必须存在！
            nn.Linear(64, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_dyn = x[:, 0:3]
        x_dist = x[:, 3:9]
        out_dyn = self.dynamics_branch(x_dyn)
        out_dist = self.dist_branch(x_dist)
        combined = torch.cat((out_dyn, out_dist), dim=1)
        logits = self.fusion_layer(combined)
        return self.sigmoid(logits)

# --- 2. 增强版 FGSM 攻击函数 ---
def fgsm_attack_batch(model, data, target, epsilon):
    """
    生成对抗样本
    """
    # 允许计算梯度
    data.requires_grad = True
    
    # 前向传播
    output = model(data)
    
    # 计算 Loss (我们希望 Loss 变大，即预测结果偏离 Target)
    loss = nn.BCELoss()(output, target)
    
    # 梯度归零并反向传播
    model.zero_grad()
    loss.backward()
    
    # 【诊断】检查梯度是否消失
    grad_mag = data.grad.data.abs().mean().item()
    if grad_mag < 1e-6:
        # 如果梯度太小，说明模型太自信，或者处于饱和区，推不动
        # 这种情况下直接返回原数据，避免报错
        return data.detach()
    
    # 获取梯度方向
    data_grad = data.grad.data.sign()
    
    # 生成对抗样本: x_adv = x + epsilon * sign(grad)
    perturbed_data = data + epsilon * data_grad
    
    # 返回并切断梯度追踪 (节省内存)
    return perturbed_data.detach()

# --- 3. 主流程 ---
def main():
    print(f"[*] 正在初始化环境 (Device: {DEVICE})...")
    
    # 1. 检查文件
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[!] 错误: 模型或标准化器文件不存在。")
        return

    # 2. 加载 Scaler
    scaler = joblib.load(SCALER_PATH)
    
    # 3. 读取数据
    print(f"[*] 读取 DDoS 数据: {DDOS_FILE}")
    try:
        df_ddos = pd.read_csv(DDOS_FILE)
    except Exception as e:
        print(f"[!] 读取失败: {e}")
        return

    # 为了快速搜索最佳 Epsilon，我们先只取前 5000 条做实验
    # (如果想生成全量对抗样本，可以把 .head(5000) 去掉)
    SAMPLE_SIZE = 5000
    df_sample = df_ddos.head(SAMPLE_SIZE).copy()
    print(f"[*] 使用 {len(df_sample)} 条样本进行攻击强度测试...")
    
    feature_cols = [
        'Rate', 'Rate_Accel', 'Rate_CV', 
        'Entropy', 'Ent_Change', 'Ent_MA', 
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA'
    ]
    
    # 准备 Tensor 数据
    X_ddos = df_sample[feature_cols].values
    y_ddos = np.ones(len(df_sample)) # 真实标签全是 1 (DDoS)
    
    # 标准化 (非常重要)
    X_ddos_norm = scaler.transform(X_ddos)
    
    X_tensor = torch.tensor(X_ddos_norm, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y_ddos, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    # 4. 加载模型
    model = DPG_Net().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except RuntimeError as e:
        print(f"\n[!] 模型加载失败: {e}")
        print("[提示] 请确保代码里的 class DPG_Net 定义与训练时完全一致 (特别是 Dropout 层)。")
        return
        
    # 设为评估模式 (这就关闭了 Dropout 的随机性，但保留了层结构)
    model.eval()
    
    # 5. 基准测试 (Clean Accuracy)
    print("\n[*] --- 基准测试 (攻击前) ---")
    with torch.no_grad():
        clean_out = model(X_tensor)
        clean_acc = ((clean_out > 0.5).float().eq(y_tensor).sum() / y_tensor.shape[0]).item()
        print(f"    原始 DDoS 识别率: {clean_acc*100:.2f}%")

    # 6. 暴力搜索最佳 Epsilon
    print(f"\n[*] --- 开始暴力搜索最佳攻击强度 (FGSM) ---")
    print(f"{'Epsilon':<10} | {'Model Acc':<15} | {'Success Rate':<15}")
    print("-" * 45)
    
    best_epsilon = 0
    lowest_acc = 1.0
    best_adv_tensor = None
    
    for eps in EPSILONS_TO_SEARCH:
        # 必须 clone，否则上一轮的修改会影响下一轮
        X_temp = X_tensor.clone().detach()
        y_temp = y_tensor.clone().detach()
        
        # 执行攻击
        X_adv = fgsm_attack_batch(model, X_temp, y_temp, eps)
        
        # 测试攻击效果
        with torch.no_grad():
            out = model(X_adv)
            # 统计依然被识别为 DDoS (1) 的比例
            acc = ((out > 0.5).float().eq(y_temp).sum() / y_temp.shape[0]).item()
            success_rate = 1.0 - acc
            
            print(f"{eps:<10} | {acc*100:.2f}%          | {success_rate*100:.2f}%")
            
            # 记录最强攻击 (让准确率最低的那个)
            # 这里的逻辑是：我们要找一个让模型稍微“瞎”一点，但不要完全崩坏的参数
            # 如果想要最强样本，就找 Acc 最低的
            if acc < lowest_acc:
                lowest_acc = acc
                best_epsilon = eps
                best_adv_tensor = X_adv

    # 7. 保存结果
    if best_adv_tensor is not None:
        print("\n" + "="*45)
        print(f"[*] 搜索结束！最有效的攻击强度是 Epsilon = {best_epsilon}")
        print(f"[*] 此时模型识别率降至: {lowest_acc*100:.2f}%")
        
        # 逆标准化，还原成物理数值
        X_adv_np = best_adv_tensor.cpu().numpy()
        X_adv_original = scaler.inverse_transform(X_adv_np)
        
        # 保存为 CSV
        df_adv = pd.DataFrame(X_adv_original, columns=feature_cols)
        df_adv['Label'] = 1 # 依然标记为 DDoS，用于后续对抗训练
        
        df_adv.to_csv(ADV_OUTPUT_CSV, index=False)
        print(f"[*] 已保存 {len(df_adv)} 条对抗样本至: {ADV_OUTPUT_CSV}")
        print(f"[*] 您可以将这些数据加入训练集，进行对抗训练 (Adversarial Training)。")
    else:
        print("\n[!] 警告: 所有攻击似乎都无效 (Acc 依然很高)。")

if __name__ == "__main__":
    main()