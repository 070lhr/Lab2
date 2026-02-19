import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 全局配置 =================
DPG_MODEL_PATH = './dpg_net_model.pth'
BASELINE_MODEL_PATH = './baseline_mlp_model.pth'
SCALER_PATH = './dpg_scaler.pkl'


# 请确认这里是您实际训练时用的文件
FLASH_FILE = './flash_event_9dim_full.csv' 
DDOS_FILE = './ciciot_ddos_9dim_full.csv'

PLOT_OUTPUT = './robustness_curve_ablation.png'

# PGD 攻击参数
EPSILONS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
ALPHA = 0.5
STEPS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===============================================

# ================= 2. 模型定义 =================
# (1) 您的骄傲：DPG-Net (带双流和隐式特征丢弃的鲁棒模型)
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
        out_dyn = self.dynamics_branch(x[:, 0:3])
        out_dist = self.dist_branch(x[:, 3:9])
        return self.fusion_layer(torch.cat((out_dyn, out_dist), dim=1))

    def forward(self, x):
        return self.sigmoid(self.forward_logits(x))

# (2) 靶子模型：普通的标准 MLP (没有任何防御机制)
class Baseline_MLP(nn.Module):
    def __init__(self):
        super(Baseline_MLP, self).__init__()
        # 直接 9 维输入，不分流，没有 Dropout
        self.net = nn.Sequential(
            nn.Linear(9, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward_logits(self, x):
        return self.net(x)

    def forward(self, x):
        return self.sigmoid(self.forward_logits(x))

# ================= 3. 辅助函数 =================
def train_baseline_if_needed():
    """如果基线模型不存在，则自动使用相同的数据和 Scaler 进行训练"""
    if os.path.exists(BASELINE_MODEL_PATH):
        print("[*] 基线模型已存在，直接加载。")
        return

    print("\n[*] 未检测到基线模型，正在快速训练 Baseline MLP...")
    scaler = joblib.load(SCALER_PATH) # 必须用同一个 Scaler 保证公平
    
    df_flash = pd.read_csv(FLASH_FILE)
    df_ddos = pd.read_csv(DDOS_FILE)
    
    feature_cols = [c for c in df_flash.columns if c not in ['Label', 'timestamp', 'Unnamed: 0']]
    
    X = np.concatenate([df_flash[feature_cols].values, df_ddos[feature_cols].values], axis=0)
    y = np.concatenate([np.zeros(len(df_flash)), np.ones(len(df_ddos))], axis=0)
    
    # 使用已经 fit 好的 scaler 进行 transform
    X_norm = scaler.transform(X)
    
    dataset = TensorDataset(torch.tensor(X_norm, dtype=torch.float32), 
                            torch.tensor(y, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = Baseline_MLP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(15): # 简单训练 15 轮足以收敛
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model.forward_logits(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}/15, Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), BASELINE_MODEL_PATH)
    print(f"[*] 基线模型训练完成并保存至: {BASELINE_MODEL_PATH}\n")

def pgd_attack(model, data, target, epsilon, alpha, steps):
    """标准的 PGD 迭代攻击"""
    if epsilon == 0.0:
        return data # Epsilon为0即为干净数据测试
        
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
        
        delta = torch.clamp(data_adv - data, -epsilon, epsilon)
        data_adv = (data + delta).detach()
            
    return data_adv

# ================= 4. 主干评估与绘图 =================
def main():
    # 1. 准备基线模型
    train_baseline_if_needed()
    
    # 2. 加载两个模型
    dpg_model = DPG_Net().to(DEVICE)
    dpg_model.load_state_dict(torch.load(DPG_MODEL_PATH, map_location=DEVICE))
    dpg_model.eval()
    
    base_model = Baseline_MLP().to(DEVICE)
    base_model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=DEVICE))
    base_model.eval()
    
    # 3. 准备测试数据 (取 2000 个 DDoS 样本作为攻击目标)
    print("[*] 正在加载测试数据...")
    scaler = joblib.load(SCALER_PATH)
    df_ddos = pd.read_csv(DDOS_FILE).head(2000)
    feature_cols = [c for c in df_ddos.columns if c not in ['Label', 'timestamp', 'Unnamed: 0']]
    
    X_tensor = torch.tensor(scaler.transform(df_ddos[feature_cols].values), dtype=torch.float32).to(DEVICE)
    y_tensor = torch.ones(len(df_ddos), 1, dtype=torch.float32).to(DEVICE)
    
    # 4. 开始双线测试
    dpg_acc_list = []
    base_acc_list = []
    
    print("\n[*] 开始 PGD 对抗攻击消融测试...")
    print(f"{'Epsilon':<10} | {'DPG-Net Acc':<15} | {'Baseline Acc':<15}")
    print("-" * 45)
    
    for eps in EPSILONS:
        # 测试 DPG-Net
        adv_dpg = pgd_attack(dpg_model, X_tensor, y_tensor, eps, ALPHA, STEPS)
        with torch.no_grad():
            acc_dpg = (dpg_model(adv_dpg) > 0.5).float().eq(y_tensor).float().mean().item()
            
        # 测试 Baseline
        adv_base = pgd_attack(base_model, X_tensor, y_tensor, eps, ALPHA, STEPS)
        with torch.no_grad():
            acc_base = (base_model(adv_base) > 0.5).float().eq(y_tensor).float().mean().item()
            
        dpg_acc_list.append(acc_dpg * 100)
        base_acc_list.append(acc_base * 100)
        
        print(f"{eps:<10} | {acc_dpg*100:>6.2f}%        | {acc_base*100:>6.2f}%")

    # 5. 绘制顶级会议风格折线图
    print("\n[*] 正在绘制鲁棒性衰减曲线...")
    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    # 画线
    plt.plot(EPSILONS, dpg_acc_list, marker='o', markersize=8, linewidth=2.5, 
             color='#1f77b4', label='DPG-Net (Ours, w/ Feature Dropout)')
    plt.plot(EPSILONS, base_acc_list, marker='s', markersize=8, linewidth=2.5, 
             color='#d62728', linestyle='--', label='Baseline MLP (Standard)')
    
    # 图表修饰
    plt.title('Robustness Degradation under PGD Attack', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Perturbation Budget ($\epsilon$)', fontsize=14)
    plt.ylabel('Detection Accuracy on DDoS (%)', fontsize=14)
    
    plt.ylim(-5, 105)
    plt.xlim(min(EPSILONS) - 0.5, max(EPSILONS) + 0.5)
    plt.xticks(EPSILONS, fontsize=12)
    plt.yticks(fontsize=12)
    
    # 图例设置
    plt.legend(loc='lower left', fontsize=12, frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT)
    print(f"[*] 绘图成功！图片已保存至: {PLOT_OUTPUT}")

if __name__ == "__main__":
    main()