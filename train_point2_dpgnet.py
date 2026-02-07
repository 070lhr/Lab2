import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib # 用于保存标准化参数
import os

# ================= 配置区域 =================
# 1. 数据文件路径
# Flash 数据 (Label=0)
FLASH_FILE = './flash_event_9dim_final.csv'
# 终极 DDoS 数据 (Label=1, 含原始+降速+对抗样本)
DDOS_FILE = './ciciot_ddos_ultimate_train.csv'

# 2. 输出路径
MODEL_SAVE_PATH = './dpg_net_model.pth'
SCALER_SAVE_PATH = './dpg_scaler.pkl'

# 3. 训练超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.0005  # 较低的学习率有助于稳定收敛
EPOCHS = 50
DROPOUT_RATE = 0.5      # Feature Dropout 的概率 (50% 的概率丢弃分布特征)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

# --- 1. 定义数据集类 ---
class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 2. 核心模型: DPG-Net (带 Feature Dropout) ---
class DPG_Net(nn.Module):
    def __init__(self):
        super(DPG_Net, self).__init__()
        
        # === 分支 A: 动力学流 (Dynamics Stream) ===
        # 输入: Rate, Rate_Accel, Rate_Vol (3维)
        # 这一路是"硬核"特征，很难被欺骗
        self.dynamics_branch = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # === 分支 B: 分布流 (Distribution Stream) ===
        # 输入: Entropy(3), Size(3) (6维)
        # 这一路容易被欺骗 (对抗攻击点)
        self.dist_branch = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # === 融合层 ===
        self.fusion_layer = nn.Sequential(
            nn.Linear(16 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # 常规 Dropout 防止过拟合
            nn.Linear(64, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, training_mode=False):
        """
        x: 输入特征 (Batch, 9)
        training_mode: 是否处于训练状态 (控制 Feature Dropout)
        """
        
        # === 核心防御机制: Feature Dropout ===
        # 在训练时，随机把"分布流特征" (第3列到第8列) 全部抹零
        # 强迫模型只靠"动力学流"去判断，防止模型过度依赖 Size_Std
        if training_mode and torch.rand(1).item() < DROPOUT_RATE:
            # 创建掩码: 动力学(1), 分布(0)
            # clone() 很重要，防止修改原数据
            mask = torch.ones_like(x)
            mask[:, 3:] = 0 
            x = x * mask
        
        # 切分特征
        # 0-2: Rate, Rate_Accel, Rate_Vol
        x_dyn = x[:, 0:3]
        
        # 3-8: Entropy..., Size...
        x_dist = x[:, 3:9]
        
        # 双流并行处理
        out_dyn = self.dynamics_branch(x_dyn)
        out_dist = self.dist_branch(x_dist)
        
        # 拼接
        combined = torch.cat((out_dyn, out_dist), dim=1)
        
        # 融合决策
        logits = self.fusion_layer(combined)
        return self.sigmoid(logits)

# --- 3. 数据准备函数 ---
def load_and_preprocess():
    print("[*] 正在加载数据集...")
    
    # 读取 CSV
    try:
        df_flash = pd.read_csv(FLASH_FILE)
        df_ddos = pd.read_csv(DDOS_FILE)
    except FileNotFoundError as e:
        print(f"[!] 错误: 找不到文件 {e.filename}。请先运行数据生成脚本。")
        exit(1)

    # 确保列顺序绝对一致
    feature_cols = [
        'Rate', 'Rate_Accel', 'Rate_Vol',       # 动力学
        'Entropy', 'Ent_Change', 'Ent_MA',      # 熵
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA' # 载荷
    ]
    
    # 提取特征和标签
    X_flash = df_flash[feature_cols].values
    y_flash = np.zeros(len(df_flash)) # Flash = 0
    
    X_ddos = df_ddos[feature_cols].values
    y_ddos = np.ones(len(df_ddos))    # DDoS = 1
    
    # 合并
    X = np.concatenate([X_flash, X_ddos], axis=0)
    y = np.concatenate([y_flash, y_ddos], axis=0)
    
    print(f"    - Flash 样本数: {len(X_flash)}")
    print(f"    - DDoS 样本数: {len(X_ddos)} (含 简单/降速/对抗)")
    
    # 切分训练集和测试集 (8:2)
    # Stratify 保证测试集里也有对抗样本
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    
    # === 标准化 (StandardScaler) ===
    # 极为重要！必须保存这个 scaler 给后面的对抗测试脚本用
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 保存 Scaler
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"[*] 标准化器已保存至: {SCALER_SAVE_PATH}")
    
    return X_train, X_test, y_train, y_test

# --- 4. 训练主流程 ---
def train_model():
    # 准备数据
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    train_ds = TrafficDataset(X_train, y_train)
    test_ds = TrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    print(f"[*] 初始化 DPG-Net (Device: {DEVICE})...")
    print(f"[*] 启用动态 Feature Dropout (Rate={DROPOUT_RATE}) 以增强鲁棒性")
    
    model = DPG_Net().to(DEVICE)
    criterion = nn.BCELoss()
    # 使用 AdamW 优化器，带权重衰减，进一步防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    print("\n[*] 开始训练...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 【关键】开启 training_mode=True 以触发 Feature Dropout
            outputs = model(inputs, training_mode=True)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 打印日志
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n[*] 模型已保存至: {MODEL_SAVE_PATH}")
    
    return model, test_loader

# --- 5. 评估流程 ---
def evaluate_model(model, test_loader):
    print("\n[*] 正在评估常规测试集...")
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            
            # 测试时关闭 Feature Dropout (training_mode=False)
            outputs = model(inputs, training_mode=False)
            predicted = (outputs > 0.5).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # 打印详细报告
    print("="*60)
    print("DPG-Net Evaluation Report")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=['Flash Event', 'DDoS Attack'], digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n[混淆矩阵]")
    print(f"TN (Flash正确): {tn} | FP (Flash误报): {fp}")
    print(f"FN (DDoS漏报): {fn} | TP (DDoS拦截): {tp}")
    
    if fn > 0:
        print(f"\n[注意] 依然有 {fn} 个样本漏报，但在混合对抗训练下，这是鲁棒性的代价。")
    else:
        print("\n[完美] 即使在包含对抗样本的测试集中，模型依然表现出色！")

if __name__ == "__main__":
    trained_model, test_data_loader = train_model()
    evaluate_model(trained_model, test_data_loader)