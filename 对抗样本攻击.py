import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import sys

# ================= 配置 =================
MODEL_PATH = './dpg_net_model.pth'
SCALER_PATH = './dpg_scaler.pkl'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
ADV_OUTPUT_CSV = './adversarial_ddos_samples.csv'

# PGD 攻击配置
EPSILON = 5.0        # 最大允许扰动 (Z-score 空间，5.0 已经很大了)
ALPHA = 0.2          # 每一步走的距离 (步长)
STEPS = 40           # 迭代次数 (走的步数)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================

# --- 1. 模型定义 (保持完全一致) ---
class DPG_Net(nn.Module):
    def __init__(self):
        super(DPG_Net, self).__init__()
        self.dynamics_branch = nn.Sequential(
            nn.Linear(3, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.dist_branch = nn.Sequential(
            nn.Linear(6, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(16 + 32, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward_logits(self, x):
        x_dyn = x[:, 0:3]
        x_dist = x[:, 3:9]
        out_dyn = self.dynamics_branch(x_dyn)
        out_dist = self.dist_branch(x_dist)
        combined = torch.cat((out_dyn, out_dist), dim=1)
        logits = self.fusion_layer(combined)
        return logits

    def forward(self, x):
        return self.sigmoid(self.forward_logits(x))

# --- 2. PGD 攻击函数 (Iterative) ---
def pgd_attack(model, data, target, epsilon, alpha, steps):
    """
    PGD: 迭代式攻击，威力远超 FGSM
    """
    # 1. 随机初始化 (Random Start): 在原始点附近随机跳一下，避免陷入局部最优
    # data_adv = data.clone().detach() + torch.empty_like(data).uniform_(-epsilon, epsilon)
    # data_adv = torch.clamp(data_adv, min=data-epsilon, max=data+epsilon).detach()
    
    # 这里为了演示清晰，我们直接从原点开始 (Standard PGD)
    data_adv = data.clone().detach()
    
    criterion = nn.BCEWithLogitsLoss()
    
    for i in range(steps):
        data_adv.requires_grad = True
        
        # 前向传播 (Logits)
        logits = model.forward_logits(data_adv)
        loss = criterion(logits, target)
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 获取梯度方向
        grad = data_adv.grad.detach().sign()
        
        # 更新数据: data + alpha * sign(grad)
        data_adv = data_adv + alpha * grad
        
        # 投影 (Projection): 确保扰动不超过 epsilon
        # 计算总扰动
        delta = data_adv - data
        # 截断扰动到 [-epsilon, +epsilon] 之间
        delta = torch.clamp(delta, -epsilon, epsilon)
        
        # 应用截断后的扰动
        data_adv = (data + delta).detach()
        
        # (可选) 打印中间过程 loss，看看是不是在上升
        # if i % 10 == 0:
        #     print(f"    Step {i}/{steps}, Loss: {loss.item():.4f}")
            
    return data_adv

# --- 3. 主程序 ---
def main():
    print(f"[*] 初始化 PGD 攻击 (Steps={STEPS}, Epsilon={EPSILON}, Alpha={ALPHA})...")
    
    if not os.path.exists(MODEL_PATH): return
    scaler = joblib.load(SCALER_PATH)
    
    # 读取数据 (这次可以多读点，或者全部读)
    # 为了测试速度，先读 2000 条看看效果
    df_ddos = pd.read_csv(DDOS_FILE).head(2000)
    
    feature_cols = [
        'Rate', 'Rate_Accel', 'Rate_CV', 
        'Entropy', 'Ent_Change', 'Ent_MA', 
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA'
    ]
    
    X = df_ddos[feature_cols].values
    y = np.ones(len(df_ddos)) 
    
    X_norm = scaler.transform(X)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    model = DPG_Net().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 1. 基准测试
    with torch.no_grad():
        clean_out = model(X_tensor)
        clean_acc = ((clean_out > 0.5).float().eq(y_tensor).sum() / y_tensor.shape[0]).item()
        print(f"[*] 原始准确率: {clean_acc*100:.2f}%")

    # 2. 执行 PGD 攻击
    print(f"\n[*] 开始执行 PGD 迭代攻击...")
    X_adv = pgd_attack(model, X_tensor, y_tensor, epsilon=EPSILON, alpha=ALPHA, steps=STEPS)
    
    # 3. 评估攻击效果
    with torch.no_grad():
        adv_out = model(X_adv)
        adv_acc = ((adv_out > 0.5).float().eq(y_tensor).sum() / y_tensor.shape[0]).item()
        
        print(f"\n" + "="*40)
        print(f"[*] PGD 攻击后准确率: {adv_acc*100:.2f}%")
        print(f"[*] 攻击成功率: {(1-adv_acc)*100:.2f}%")
        
        # 看看是不是真的只有那 1.4% 的人在变
        flipped_cnt = ((adv_out > 0.5).float().eq(y_tensor) == False).sum().item()
        print(f"[*] 成功欺骗样本数: {flipped_cnt} / {len(df_ddos)}")
        print("="*40)

    # 4. 保存对抗样本 (只保存成功的，或者全部保存？通常对抗训练需要全部保存)
    # 我们全部保存，即使没成功的，也是“最难识别”的 DDoS
    X_adv_np = X_adv.cpu().numpy()
    X_ori = scaler.inverse_transform(X_adv_np)
    
    df_adv = pd.DataFrame(X_ori, columns=feature_cols)
    df_adv['Label'] = 1 
    
    df_adv.to_csv(ADV_OUTPUT_CSV, index=False)
    print(f"[*] 对抗样本已保存至: {ADV_OUTPUT_CSV}")

if __name__ == "__main__":
    main()