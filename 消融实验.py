#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
FLASH_FILE = './flash_event_9dim_full.csv' 
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
# ===========================================

# ---------------------------------------------------------
# 1. 核心消融变体定义 (严格控制变量)
# ---------------------------------------------------------

# 变体 A: 仅使用 6 维静态分布特征 (Dist-Only)
class Dist_Only_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward_logits(self, x): return self.net(x[:, 3:9]) # 仅截取后 6 维
    def forward(self, x): return torch.sigmoid(self.forward_logits(x))

# 变体 B: 仅使用 3 维动力学时序特征 (Dyn-Only)
class Dyn_Only_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward_logits(self, x): return self.net(x[:, 0:3]) # 仅截取前 3 维
    def forward(self, x): return torch.sigmoid(self.forward_logits(x))

# 变体 C: 9 维全特征，但去除双流隔离与 Dropout (No-Dropout / Linear MLP)
class No_Dropout_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1) # 直接输出，没有任何隐式丢弃保护
        )
    def forward_logits(self, x): return self.net(x)
    def forward(self, x): return torch.sigmoid(self.forward_logits(x))

# 满血版 D: 您的终极双流架构 (DPG-Net)
class DPG_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamics_branch = nn.Sequential(nn.Linear(3, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.dist_branch = nn.Sequential(nn.Linear(6, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.fusion_layer = nn.Sequential(
            nn.Linear(16 + 32, 64), nn.ReLU(),
            nn.Dropout(0.3), # 核心抗扰动机制
            nn.Linear(64, 1)
        )
    def forward_logits(self, x):
        out_dyn = self.dynamics_branch(x[:, 0:3])
        out_dist = self.dist_branch(x[:, 3:9])
        return self.fusion_layer(torch.cat((out_dyn, out_dist), dim=1))
    def forward(self, x): return torch.sigmoid(self.forward_logits(x))

# ---------------------------------------------------------
# 2. PGD 攻击核心函数
# ---------------------------------------------------------
def pgd_attack(model, data, target, epsilon, alpha, steps=40):
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
        delta = torch.clamp(data_adv - data, min=-epsilon, max=epsilon)
        data_adv = (data + delta).detach()
    return data_adv

# ---------------------------------------------------------
# 3. 统一化训练与评估流
# ---------------------------------------------------------
def train_and_eval_ablation(model_name, model, train_loader, X_ddos, y_ddos, X_fe, y_fe):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练模型
    model.train()
    for epoch in range(EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model.forward_logits(inputs), labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    
    # --- 评估 1：洁净环境 (Eps=0.0) ---
    X_ddos_tensor = torch.tensor(X_ddos, dtype=torch.float32).to(DEVICE)
    y_ddos_tensor = torch.tensor(y_ddos, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_fe_tensor = torch.tensor(X_fe, dtype=torch.float32).to(DEVICE)
    
    X_mixed_clean = torch.cat((X_ddos_tensor, X_fe_tensor), dim=0)
    y_mixed_true = np.concatenate((y_ddos, y_fe))
    
    with torch.no_grad():
        preds_clean = (model(X_mixed_clean) > 0.5).float().cpu().numpy().flatten()
    
    # 计算洁净环境的四大指标
    acc_clean = accuracy_score(y_mixed_true, preds_clean)
    prec_clean = precision_score(y_mixed_true, preds_clean, zero_division=0)
    rec_clean = recall_score(y_mixed_true, preds_clean, zero_division=0)
    f1_clean = f1_score(y_mixed_true, preds_clean, zero_division=0)
    
    # --- 评估 2：极限对抗环境 (Eps=0.8) ---
    alpha = 0.8 / 5.0
    X_adv_tensor = pgd_attack(model, X_ddos_tensor, y_ddos_tensor, epsilon=0.8, alpha=alpha, steps=40)
    X_mixed_adv = torch.cat((X_adv_tensor, X_fe_tensor), dim=0)
    
    with torch.no_grad():
        preds_adv = (model(X_mixed_adv) > 0.5).float().cpu().numpy().flatten()
        
    # 计算极限环境的四大指标
    acc_adv = accuracy_score(y_mixed_true, preds_adv)
    prec_adv = precision_score(y_mixed_true, preds_adv, zero_division=0)
    rec_adv = recall_score(y_mixed_true, preds_adv, zero_division=0)
    f1_adv = f1_score(y_mixed_true, preds_adv, zero_division=0)
    
    # --- 优雅的日志输出 ---
    print(f"[{model_name}]")
    print(f"  -> 洁净环境 (Eps=0.0): Acc={acc_clean*100:6.2f}% | Pre={prec_clean*100:6.2f}% | Rec={rec_clean*100:6.2f}% | F1={f1_clean*100:6.2f}%")
    print(f"  -> 极限攻击 (Eps=0.8): Acc={acc_adv*100:6.2f}% | Pre={prec_adv*100:6.2f}% | Rec={rec_adv*100:6.2f}% | F1={f1_adv*100:6.2f}%")
    print("-" * 80)

def main():
    print("[*] 正在加载双路数据并构建完全对等的 1:1 实验场...")
    try:
        df_flash = pd.read_csv(FLASH_FILE)
        df_ddos = pd.read_csv(DDOS_FILE)
    except FileNotFoundError:
        print("[!] 找不到数据文件，请检查路径。")
        return

    df_flash['Label'] = 0.0
    df_ddos['Label'] = 1.0
    
    min_samples = min(len(df_flash), len(df_ddos))
    df_fe_sampled = df_flash.sample(n=min_samples, random_state=42)
    df_ddos_sampled = df_ddos.sample(n=min_samples, random_state=42)
    df_all = pd.concat([df_fe_sampled, df_ddos_sampled])
    
    feature_cols = ['Size_Std', 'SizeStd_Change', 'SizeStd_MA', 'SIP_Ent',  'SIPEnt_MA', 'SIPEnt_Change', 'Rate', 'Rate_Accel', 'Rate_CV']
    X = df_all[feature_cols].values
    y = df_all['Label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    ddos_mask = (y_test == 1.0)
    fe_mask = (y_test == 0.0)
    X_test_ddos, y_test_ddos = X_test[ddos_mask], y_test[ddos_mask]
    X_test_fe, y_test_fe = X_test[fe_mask], y_test[fe_mask]

    print("\n" + "="*80)
    print(" 🔬 4.5.4 架构消融实验结果 (控制变量法 + 全指标) 🔬")
    print("="*80)
    
    variants = [
        ("Dist-Only (单流-仅分布)", Dist_Only_Net()),
        ("Dyn-Only (单流-仅动力学)", Dyn_Only_Net()),
        ("No-Dropout (无隔离朴素融合)", No_Dropout_Net()),
        ("DPG-Net (Ours 满血双流)", DPG_Net())
    ]
    
    for name, model in variants:
        train_and_eval_ablation(name, model, train_loader, X_test_ddos, y_test_ddos, X_test_fe, y_test_fe)
        
    print("[+] 实验完成！")

if __name__ == "__main__":
    main()