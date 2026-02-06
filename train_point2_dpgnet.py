import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ================= 配置 =================
# 数据文件
FLASH_FILE = './flash_event_9dim_ready.csv'    # Label = 0
DDOS_FILE = './ciciot_ddos_9dim_final.csv'     # Label = 1 (含原始+削弱)
MODEL_PATH = './dpg_net_model.pth'

# 训练超参数
BATCH_SIZE = 64
LR = 0.0005        # 稍微调低学习率，让双流网络学得更稳
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================

# 1. 定义数据集
class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. 核心创新模型：双流门控网络 (DPG-Net)
class DPG_Net(nn.Module):
    def __init__(self):
        super(DPG_Net, self).__init__()
        
        # === 分支 A: 动力学流 (Dynamics Stream) ===
        # 输入: Rate, Rate_Accel, Rate_Vol (3维)
        self.dynamics_branch = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # === 分支 B: 分布流 (Distribution Stream) ===
        # 输入: Entropy, Ent_Change, Ent_MA, Size_Std, Size_Change, Size_MA (6维)
        self.dist_branch = nn.Sequential(
            nn.Linear(6, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # === 融合门控层 (Gated Fusion) ===
        # 将两个分支的特征拼接 (8 + 16 = 24)
        self.fusion_layer = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Dropout(0.3), # 防止过拟合
            nn.Linear(32, 1) # 输出 Logits
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 的维度是 (Batch, 9)
        # 按照列顺序切分特征
        # Dynamics: Rate(0), Rate_Accel(1), Rate_Vol(2)
        x_dynamics = x[:, 0:3]
        
        # Distribution: Entropy(3,4,5), Size(6,7,8)
        x_dist = x[:, 3:9]
        
        # 分别通过两个分支
        out_dyn = self.dynamics_branch(x_dynamics) # -> (Batch, 8)
        out_dist = self.dist_branch(x_dist)        # -> (Batch, 16)
        
        # 拼接
        combined = torch.cat((out_dyn, out_dist), dim=1)
        
        # 融合分类
        logits = self.fusion_layer(combined)
        return self.sigmoid(logits)

# 3. 数据加载与预处理
def prepare_data():
    print("[*] 正在加载并预处理数据...")
    
    # 读取
    df_flash = pd.read_csv(FLASH_FILE)
    df_ddos = pd.read_csv(DDOS_FILE)
    
    # 确保列顺序一致 (非常重要，对应模型里的切片)
    feature_cols = [
        'Rate', 'Rate_Accel', 'Rate_Vol',       # 0-2: 动力学
        'Entropy', 'Ent_Change', 'Ent_MA',      # 3-5: 熵
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA' # 6-8: 载荷
    ]
    
    # 提取 X 和 y
    X_flash = df_flash[feature_cols].values
    y_flash = df_flash['Label'].values
    
    X_ddos = df_ddos[feature_cols].values
    y_ddos = df_ddos['Label'].values
    
    # 合并
    X = np.concatenate([X_flash, X_ddos], axis=0)
    y = np.concatenate([y_flash, y_ddos], axis=0)
    
    print(f"    - Flash 样本: {len(X_flash)}")
    print(f"    - DDoS 样本: {len(X_ddos)} (含隐蔽样本)")
    
    # 切分训练/测试集
    # stratify=y 保证了两边都有比例一致的 Flash 和 DDoS (含强/弱)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 标准化 (StandardScaler)
    # 注意：动力学特征数值很大(几千)，熵很小(几)，必须标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# 4. 训练函数
def train():
    X_train, X_test, y_train, y_test = prepare_data()
    
    # 转 DataSet
    train_ds = TrafficDataset(X_train, y_train)
    test_ds = TrafficDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化
    print(f"[*] 初始化 DPG-Net 模型 (Device: {DEVICE})...")
    model = DPG_Net().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) # AdamW 抗过拟合更好
    
    # 训练循环
    print("[*] 开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 每 10 轮打印一次
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")
            
    # 保存
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[*] 训练完成，模型保存至: {MODEL_PATH}")
    
    return model, test_loader

# 5. 评估函数
def evaluate(model, test_loader):
    print("\n[*] 正在评估测试集 (含隐蔽攻击样本)...")
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            preds = (outputs > 0.5).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    # 输出报告
    print("\n" + "="*60)
    print("DPG-Net Evaluation Report (Flash Event vs. Stealthy DDoS)")
    print("="*60)
    
    print(classification_report(y_true, y_pred, 
                                target_names=['Flash Event (Normal)', 'DDoS Attack (Malicious)'], 
                                digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n[混淆矩阵详解]")
    print(f"真阴性 (TN): {tn} (Flash 被正确识别)")
    print(f"假阳性 (FP): {fp} (Flash 被误报为攻击) -> 误报率")
    print(f"假阴性 (FN): {fn} (DDoS 被漏判为正常) -> 漏报率 (最危险)")
    print(f"真阳性 (TP): {tp} (DDoS 被正确拦截)")

if __name__ == "__main__":
    trained_model, test_loader = train()
    evaluate(trained_model, test_loader)