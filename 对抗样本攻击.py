import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os

# ================= 配置 =================
MODEL_PATH = './dpg_net_model.pth'
SCALER_PATH = './dpg_scaler.pkl'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 极限测试配置
# Epsilon: 允许的最大标准差偏移量 (10.0 已经极度离谱，200.0 是神仙难救)
EXTREME_EPSILONS = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
STEPS = 50  # 增加迭代次数，确保在高 Epsilon 下能跑到边界
# =======================================

# --- 1. 模型定义 (保持原样) ---
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
        return self.fusion_layer(combined)

    def forward(self, x):
        return self.sigmoid(self.forward_logits(x))

# --- 2. PGD 核心函数 ---
def pgd_attack(model, data, target, epsilon, alpha, steps):
    data_adv = data.clone().detach()
    criterion = nn.BCEWithLogitsLoss()
    
    for _ in range(steps):
        data_adv.requires_grad = True
        logits = model.forward_logits(data_adv)
        loss = criterion(logits, target)
        
        model.zero_grad()
        loss.backward()
        
        grad = data_adv.grad.detach().sign()
        data_adv = data_adv + alpha * grad
        
        # 投影限制
        delta = torch.clamp(data_adv - data, min=-epsilon, max=epsilon)
        data_adv = (data + delta).detach()
            
    return data_adv

# --- 3. 主程序 ---
def main():
    print(f"[*] 初始化极限 PGD 测试 (Device: {DEVICE})...")
    
    if not os.path.exists(MODEL_PATH): return
    scaler = joblib.load(SCALER_PATH)
    
    # 取 2000 条测试以保证速度
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
    
    # 基准测试
    with torch.no_grad():
        clean_out = model(X_tensor)
        clean_acc = ((clean_out > 0.5).float().eq(y_tensor).sum() / y_tensor.shape[0]).item()
    
    print(f"\n[*] --- 开始探底测试 (原始准确率: {clean_acc*100:.2f}%) ---")
    print(f"{'Epsilon (极值)':<15} | {'Alpha (步长)':<15} | {'Model Acc':<15} | {'破防率':<15}")
    print("-" * 65)
    
    for eps in EXTREME_EPSILONS:
        # 动态调整步长：确保 50 步内能走到边界
        alpha = eps / (STEPS / 2.0) 
        
        X_adv = pgd_attack(model, X_tensor, y_tensor, epsilon=eps, alpha=alpha, steps=STEPS)
        
        with torch.no_grad():
            adv_out = model(X_adv)
            adv_acc = ((adv_out > 0.5).float().eq(y_tensor).sum() / y_tensor.shape[0]).item()
            success_rate = 1.0 - adv_acc
            
            print(f"{eps:<15} | {alpha:<15.2f} | {adv_acc*100:>8.2f}%      | {success_rate*100:>8.2f}%")
            
            if adv_acc < 0.1:
                print(f"\n[!] 模型在 Epsilon={eps} 处防线彻底崩溃。")
                break

if __name__ == "__main__":
    main()