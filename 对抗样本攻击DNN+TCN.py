#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ================= 配置区域 =================
# 您的 9 维全量特征数据集路径
FLASH_FILE = './flash_event_9dim_full.csv'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

# PGD 攻击参数 (白盒攻击极其精准，0.8的扰动足以让模型崩溃)
EPSILON = 0.8      
ALPHA = 0.1        
NUM_ITER = 40      
# ===========================================

# ---------------------------------------------------------
# 1. 模型架构定义 (DNN 与 TCN)
# ---------------------------------------------------------
class DNN_Model(nn.Module):
    def __init__(self, input_dim=9, num_classes=2):
        super(DNN_Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class TCN_Model(nn.Module):
    def __init__(self, input_dim=9, num_classes=2):
        super(TCN_Model, self).__init__()
        # 1D 卷积提取时序特征
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=2, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=4, dilation=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        x = x.unsqueeze(1) # 增加 channel 维度
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

# ---------------------------------------------------------
# 2. 白盒 PGD 对抗样本生成器
# ---------------------------------------------------------
def pgd_attack_whitebox(model, X, y, epsilon, alpha, num_iter):
    """直接对目标深度学习模型发动白盒梯度攻击"""
    model.eval() # 冻结 BatchNorm 和 Dropout，确保梯度计算准确
    criterion = nn.CrossEntropyLoss()
    X_adv = X.clone().detach()
    
    for _ in range(num_iter):
        X_adv.requires_grad_(True)
        outputs = model(X_adv)
        loss = criterion(outputs, y)
        
        model.zero_grad()
        loss.backward()
        
        grad = X_adv.grad.detach()
        X_adv = X_adv + alpha * grad.sign()
        
        # 数学投影，限制扰动幅度
        eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
        X_adv = (X + eta).detach()
        
    return X_adv

# ---------------------------------------------------------
# 3. 核心攻防流程
# ---------------------------------------------------------
def train_and_evaluate(model, model_name, train_loader, X_test, y_test, device):
    print(f"\n{'='*60}")
    print(f"[*] 开始对 {model_name} 进行训练与白盒攻防测试")
    print(f"{'='*60}")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 阶段一：模型训练 ---
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # --- 阶段二：洁净测试集评估 ---
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs_clean = model(X_test_tensor)
        _, preds_clean = torch.max(outputs_clean, 1)
    
    preds_clean_np = preds_clean.cpu().numpy()
    acc_clean = accuracy_score(y_test, preds_clean_np)
    print(f"[+] 攻击前 {model_name} 洁净测试集整体准确率 = {acc_clean*100:.2f}%")

    # --- 阶段三：发动 PGD 白盒攻击 ---
    print(f"[*] 正在对测试集中的 DDoS 流量发动白盒 PGD 拟态对抗攻击...")
    
    # 提取测试集中的 DDoS (1) 和 正常流量 (0)
    ddos_indices = np.where(y_test == 1)[0]
    X_ddos_clean = X_test[ddos_indices]
    y_ddos_true = y_test[ddos_indices]
    
    fe_indices = np.where(y_test == 0)[0]
    X_fe_clean = X_test[fe_indices]
    y_fe_true = y_test[fe_indices]
    
    X_ddos_tensor = torch.FloatTensor(X_ddos_clean).to(device)
    y_ddos_tensor = torch.LongTensor(y_ddos_true).to(device)
    
    # 针对当前模型生成对抗样本
    X_adv_tensor = pgd_attack_whitebox(model, X_ddos_tensor, y_ddos_tensor, EPSILON, ALPHA, NUM_ITER)
    X_adv_np = X_adv_tensor.cpu().numpy()
    
    # 重组受污染的全局测试集
    X_test_attacked = np.vstack((X_adv_np, X_fe_clean))
    y_test_attacked = np.hstack((y_ddos_true, y_fe_true)) 
    
    # --- 阶段四：对抗评估 ---
    X_test_attacked_tensor = torch.FloatTensor(X_test_attacked).to(device)
    with torch.no_grad():
        outputs_adv = model(X_test_attacked_tensor)
        _, preds_adv = torch.max(outputs_adv, 1)
        
    preds_adv_np = preds_adv.cpu().numpy()
    
    # 计算专门针对恶意流量的灾难性后果
    acc_adv = accuracy_score(y_test_attacked, preds_adv_np)
    prec_adv = precision_score(y_test_attacked, preds_adv_np, zero_division=0)
    rec_adv = recall_score(y_test_attacked, preds_adv_np, zero_division=0)
    f1_adv = f1_score(y_test_attacked, preds_adv_np, zero_division=0)
    
    print(f"\n 💥 {model_name} 遭遇白盒 PGD 攻击崩溃结果 💥")
    print(f"攻击扰动幅度 (Epsilon)  = {EPSILON}")
    print(f"攻击后全局 Accuracy (%)  = {acc_adv * 100:.2f}")
    print(f"攻击后恶意 Precision (%) = {prec_adv * 100:.2f}")
    print(f"攻击后恶意 Recall (%)    = {rec_adv * 100:.2f}")
    print(f"攻击后恶意 F1-Score (%)  = {f1_adv * 100:.2f}")
    print("-" * 60)

def main():
    print("[*] 正在加载并重组 9 维双路特征数据...")
    try:
        df_flash = pd.read_csv(FLASH_FILE)
        df_ddos = pd.read_csv(DDOS_FILE)
    except FileNotFoundError as e:
        print(f"[!] 文件读取失败，请检查路径: {e}")
        return

    df_flash['Label'] = 0
    df_ddos['Label'] = 1

    # 1:1 完美平衡采样，彻底消除准确率幻觉
    min_samples = min(len(df_flash), len(df_ddos))
    df_fe_sampled = df_flash.sample(n=min_samples, random_state=42)
    df_ddos_sampled = df_ddos.sample(n=min_samples, random_state=42)
    
    df = pd.concat([df_fe_sampled, df_ddos_sampled]).sample(frac=1.0, random_state=42)
    print(f"[+] 数据 1:1 平衡重组完毕！总样本数: {len(df)}")

    feature_cols = df.columns[:9] 
    X = df[feature_cols].values
    y = df['Label'].values

    # 深度学习必须进行的标准化映射
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 当前使用的计算设备为: {device}")

    # ================= 运行 DNN 攻防测试 =================
    dnn_model = DNN_Model(input_dim=9, num_classes=2)
    train_and_evaluate(dnn_model, "DNN (9D)", train_loader, X_test, y_test, device)

    # ================= 运行 TCN 攻防测试 =================
    tcn_model = TCN_Model(input_dim=9, num_classes=2)
    train_and_evaluate(tcn_model, "TCN (9D)", train_loader, X_test, y_test, device)

if __name__ == "__main__":
    main()