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
# 分别指定正常突发流量与 DDoS 攻击流量的文件路径
FLASH_FILE = './flash_event_9dim_full.csv'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
# ===========================================

# ---------------------------------------------------------
# 1. 模型架构定义
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
        x = x.unsqueeze(1) 
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

# ---------------------------------------------------------
# 2. 训练与评估流程
# ---------------------------------------------------------

def evaluate_model(model, dataloader, device, model_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"\n[{model_name}] 洁净测试集评估结果:")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")
    print(f"F1-Score  : {f1*100:.2f}%")
    print("-" * 40)
    return model

def train_model(model, train_loader, test_loader, device, model_name):
    print(f"\n[*] 开始训练 {model_name} ...")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")
            
    evaluate_model(model, test_loader, device, model_name)
    return model

def main():
    print("[*] 正在分别加载 Flash Event 与 DDoS 9维特征数据...")
    try:
        df_flash = pd.read_csv(FLASH_FILE)
        df_ddos = pd.read_csv(DDOS_FILE)
    except FileNotFoundError as e:
        print(f"[!] 文件读取失败，请检查路径: {e}")
        return

    # 动态分配物理级标签
    df_flash['Label'] = 0
    df_ddos['Label'] = 1

    # 沿纵向合并为完整的异构混合数据集
    df = pd.concat([df_flash, df_ddos], ignore_index=True)
    print(f"[+] 数据合并完成！总样本数: {len(df)} (FE: {len(df_flash)}, DDoS: {len(df_ddos)})")

    # 提取特征与标签 (这里严格选取前9列作为输入特征)
    feature_cols = df.columns[:9] 
    X = df[feature_cols].values
    y = df['Label'].values

    # 特征空间标准化映射
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 当前使用的计算设备为: {device}")

    # ================= 运行模型 =================
    dnn_model = DNN_Model(input_dim=9, num_classes=2)
    train_model(dnn_model, train_loader, test_loader, device, "DNN (9D)")

    tcn_model = TCN_Model(input_dim=9, num_classes=2)
    train_model(tcn_model, train_loader, test_loader, device, "TCN (9D)")

if __name__ == "__main__":
    main()