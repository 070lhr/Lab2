#!/usr/bin/env python3
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ================= 配置 =================
MODEL_PATH = './dpg_net_model.pth'
SCALER_PATH = './dpg_scaler.pkl'
# 请确保这两个文件路径正确，以便构成 1:1 混合测试集
FLASH_FILE = './flash_event_9dim_full.csv' 
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 对齐 4.5.3 节组图的扰动阶梯
EPSILONS = [0.0, 0.2, 0.4, 0.6, 0.8]
STEPS = 40  
# =======================================

# --- 1. 模型定义 (保持您的双流隐式架构不变) ---
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

# --- 2. PGD 核心函数 (白盒攻击) ---
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
        
        # 数学投影限制
        delta = torch.clamp(data_adv - data, min=-epsilon, max=epsilon)
        data_adv = (data + delta).detach()
            
    return data_adv

# --- 3. 主程序 ---
def main():
    print(f"[*] 初始化 DPG-Net 对抗演进测试 (Device: {DEVICE})...")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH): 
        print("[!] 找不到模型文件或归一化器，请检查路径。")
        return
        
    scaler = joblib.load(SCALER_PATH)
    
    # 加载双路数据并进行 1:1 平衡采样
    print("[*] 正在加载并重组双路测试集...")
    df_flash = pd.read_csv(FLASH_FILE)
    df_ddos = pd.read_csv(DDOS_FILE)
    
    df_flash['Label'] = 0.0
    df_ddos['Label'] = 1.0
    
    # 限制总样本数以加快 PGD 运行速度 (各取 2000 条，总计 4000 条构成完美的 1:1)
    min_samples = min(2000, len(df_flash), len(df_ddos))
    df_fe_sampled = df_flash.sample(n=min_samples, random_state=42)
    df_ddos_sampled = df_ddos.sample(n=min_samples, random_state=42)
    
    # 您指定的 9 维特征列
    feature_cols = [
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA', 
        'SIP_Ent',  'SIPEnt_MA', 'SIPEnt_Change',
        'Rate', 'Rate_Accel', 'Rate_CV', 
    ]
    
    # 提取纯 DDoS 样本用于生成对抗样本
    X_ddos = df_ddos_sampled[feature_cols].values
    y_ddos = df_ddos_sampled['Label'].values
    X_ddos_norm = scaler.transform(X_ddos)
    X_ddos_tensor = torch.tensor(X_ddos_norm, dtype=torch.float32).to(DEVICE)
    y_ddos_tensor = torch.tensor(y_ddos, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    # 提取纯 FE 样本 (不受攻击)
    X_fe = df_fe_sampled[feature_cols].values
    y_fe = df_fe_sampled['Label'].values
    X_fe_norm = scaler.transform(X_fe)
    X_fe_tensor = torch.tensor(X_fe_norm, dtype=torch.float32).to(DEVICE)
    
    # 加载模型
    model = DPG_Net().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    print("\n" + "="*70)
    print(f"{'Epsilon':<8} | {'Accuracy (%)':<13} | {'Precision (%)':<13} | {'Recall (%)':<13} | {'F1-Score (%)':<13}")
    print("-" * 70)
    
    # 阶梯测试循环
    for eps in EPSILONS:
        if eps == 0.0:
            X_adv_tensor = X_ddos_tensor # 无扰动
        else:
            # 自适应步长：确保在 40 步内能细腻地探索特征空间
            alpha = eps / 5.0 
            X_adv_tensor = pgd_attack(model, X_ddos_tensor, y_ddos_tensor, epsilon=eps, alpha=alpha, steps=STEPS)
        
        # 重新混合：将受攻击的 DDoS 与干净的 FE 拼接
        X_mixed_tensor = torch.cat((X_adv_tensor, X_fe_tensor), dim=0)
        y_mixed_true = np.concatenate((y_ddos, y_fe))
        
        with torch.no_grad():
            outputs = model(X_mixed_tensor)
            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            
        # 计算针对恶意流量的四项核心指标
        acc = accuracy_score(y_mixed_true, preds)
        prec = precision_score(y_mixed_true, preds, zero_division=0)
        rec = recall_score(y_mixed_true, preds, zero_division=0)
        f1 = f1_score(y_mixed_true, preds, zero_division=0)
        
        print(f"{eps:<8.1f} | {acc*100:<13.2f} | {prec*100:<13.2f} | {rec*100:<13.2f} | {f1*100:<13.2f}")
        
    print("="*70)
    print("[*] 专家提示：将这组坚挺的数据替换到之前的 2x2 组图代码中，即可完美展示您架构的防御纵深！")

if __name__ == "__main__":
    main()