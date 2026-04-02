import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib 
import os
import shap 
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于显示进度条
import time

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
FLASH_FILE = './flash_event_9dim_full.csv'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
MODEL_SAVE_PATH = './dpg_net_model.pth'
SCALER_SAVE_PATH = './dpg_scaler.pkl'

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 50
DROPOUT_RATE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 严格匹配论文 4.2 节定义的 9 维双流特征列名
FEATURE_COLS = [
    'Rate', 'Rate_Accel', 'Rate_CV',           # 动力学流 [cite: 384]
    'SIP_Ent', 'SIPEnt_Change', 'SIPEnt_MA',   # 分布流-熵 [cite: 361]
    'Size_Std', 'SizeStd_Change', 'SizeStd_MA' # 分布流-载荷 [cite: 361]
]
# ===========================================

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class DPG_Net(nn.Module):
    def __init__(self):
        super(DPG_Net, self).__init__()
        # 动力学分支：处理 3 维核心动力学特征 [cite: 353, 418]
        self.dynamics_branch = nn.Sequential(
            nn.Linear(3, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        # 分布分支：处理 6 维统计分布特征 [cite: 353, 407]
        self.dist_branch = nn.Sequential(
            nn.Linear(6, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        # 融合与决策层 [cite: 429]
        self.fusion_layer = nn.Sequential(
            nn.Linear(16 + 32, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, training_mode=False):
        if training_mode and torch.rand(1).item() < DROPOUT_RATE:
            mask = torch.ones_like(x)
            mask[:, 3:] = 0 # 随机遮蔽分布流特征，强化动力学特征学习 [cite: 356, 431]
            x = x * mask
        
        out_dyn = self.dynamics_branch(x[:, 0:3])
        out_dist = self.dist_branch(x[:, 3:9])
        combined = torch.cat((out_dyn, out_dist), dim=1)
        return self.sigmoid(self.fusion_layer(combined))

def load_and_preprocess():
    print("\n>>> [阶段 1/4] 数据预处理与标准化...")
    if not (os.path.exists(FLASH_FILE) and os.path.exists(DDOS_FILE)):
        print(f"  [!] 错误：找不到数据文件，请检查路径。")
        exit()

    df_flash = pd.read_csv(FLASH_FILE)
    df_ddos = pd.read_csv(DDOS_FILE)
    
    X = np.concatenate([df_flash[FEATURE_COLS].values, df_ddos[FEATURE_COLS].values], axis=0)
    y = np.concatenate([np.zeros(len(df_flash)), np.ones(len(df_ddos))], axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    print(f"  [+] 数据加载完毕。训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    train_loader = DataLoader(TrafficDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TrafficDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n>>> [阶段 2/4] DPG-Net 模型训练 (Device: {DEVICE})...")
    model = DPG_Net().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    # 使用 tqdm 封装 epoch 循环
    pbar = tqdm(range(EPOCHS), desc="训练进度")
    for epoch in pbar:
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs, training_mode=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  [+] 训练完成，模型已保存。")
    return model, test_loader, X_test

def evaluate_model(model, test_loader):
    print("\n>>> [阶段 3/4] 常规检测性能评估...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(DEVICE), training_mode=False)
            y_true.extend(labels.numpy())
            y_pred.extend((outputs > 0.5).cpu().numpy())
    
    print("-" * 30)
    print(classification_report(y_true, y_pred, target_names=['Flash', 'DDoS'], digits=4))
    print("-" * 30)

def run_shap_analysis(model, X_test):
    print("\n>>> [阶段 4/4] SHAP 解释性归因分析...")
    print("  [提示] SHAP 正在计算高维空间的特征梯度，这可能需要 1-3 分钟，请稍候...")
    model.eval()
    
    # 抽取背景样本和待解释样本
    bg_data = torch.tensor(X_test[np.random.choice(X_test.shape[0], 100, replace=False)], dtype=torch.float32).to(DEVICE)
    test_data = torch.tensor(X_test[np.random.choice(X_test.shape[0], 200, replace=False)], dtype=torch.float32).to(DEVICE)
    
    start_time = time.time()
    explainer = shap.DeepExplainer(model, bg_data)
    shap_values = explainer.shap_values(test_data)
    end_time = time.time()
    
    print(f"  [+] SHAP 计算完成，耗时: {end_time - start_time:.2f}s")

    # 绘图逻辑
    plt.rcParams['font.sans-serif'] = ['SimHei']; plt.rcParams['axes.unicode_minus'] = False
    
    print("  [*] 正在生成特征重要性柱状图...")
    plt.figure()
    shap.summary_plot(shap_values, test_data.cpu().numpy(), feature_names=FEATURE_COLS, plot_type="bar", show=False)
    plt.savefig("shap_bar_real.png", dpi=300, bbox_inches='tight')
    
    print("  [*] 正在生成决策机制蜂巢图...")
    plt.figure()
    shap.summary_plot(shap_values, test_data.cpu().numpy(), feature_names=FEATURE_COLS, show=False)
    plt.savefig("shap_beeswarm_real.png", dpi=300, bbox_inches='tight')
    
    print(f"\n[任务完成] 所有结果已保存至当前目录。祝你大论文顺利！")

if __name__ == "__main__":
    trained_model, test_loader, X_test_raw = train_model()
    evaluate_model(trained_model, test_loader)
    run_shap_analysis(trained_model, X_test_raw)