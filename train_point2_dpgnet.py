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
import shap  # 新增
import matplotlib.pyplot as plt

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

FEATURE_COLS = [
    'Rate', 'Rate_Accel', 'Rate_CV',              # 动力学特征 (0, 1, 2)
    'SIP_Ent', 'SIPEnt_Change', 'SIPEnt_MA',      # 源IP信息熵特征 (3, 4, 5) - 修改此处
    'Size_Std', 'SizeStd_Change', 'SizeStd_MA'    # 数据包载荷特征 (6, 7, 8)
]
# ===========================================

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DPG_Net(nn.Module):
    def __init__(self):
        super(DPG_Net, self).__init__()
        self.dynamics_branch = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.dist_branch = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(16 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, training_mode=False):
        # 注意：SHAP 调用时 training_mode 默认为 False，符合逻辑
        if training_mode and torch.rand(1).item() < DROPOUT_RATE:
            mask = torch.ones_like(x)
            mask[:, 3:] = 0 
            x = x * mask
        
        x_dyn = x[:, 0:3]
        x_dist = x[:, 3:9]
        
        out_dyn = self.dynamics_branch(x_dyn)
        out_dist = self.dist_branch(x_dist)
        combined = torch.cat((out_dyn, out_dist), dim=1)
        logits = self.fusion_layer(combined)
        return self.sigmoid(logits)

def load_and_preprocess():
    df_flash = pd.read_csv(FLASH_FILE)
    df_ddos = pd.read_csv(DDOS_FILE)
    X_flash = df_flash[FEATURE_COLS].values
    y_flash = np.zeros(len(df_flash))
    X_ddos = df_ddos[FEATURE_COLS].values
    y_ddos = np.ones(len(df_ddos))
    X = np.concatenate([X_flash, X_ddos], axis=0)
    y = np.concatenate([y_flash, y_ddos], axis=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    train_ds = TrafficDataset(X_train, y_train)
    test_ds = TrafficDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DPG_Net().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs, training_mode=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    return model, test_loader, X_test

# --- 6. 新增：SHAP 解释性分析函数 ---
def run_shap_analysis(model, X_test):
    print("\n[*] 开始进行 SHAP 解释性分析 (使用真实测试数据)...")
    model.eval()
    
    # 1. 准备背景数据集 (Background Distribution)
    # 从测试集中随机抽取 100 个样本作为背景，帮助 SHAP 理解“基准”
    background_idx = np.random.choice(X_test.shape[0], 100, replace=False)
    background = torch.tensor(X_test[background_idx], dtype=torch.float32).to(DEVICE)
    
    # 2. 准备待解释样本 (测试集前 200 个)
    test_idx = np.random.choice(X_test.shape[0], 200, replace=False)
    test_samples = torch.tensor(X_test[test_idx], dtype=torch.float32).to(DEVICE)
    
    # 3. 初始化 SHAP DeepExplainer
    # 注意：我们直接传 model，SHAP 会默认调用 forward(x, training_mode=False)
    explainer = shap.DeepExplainer(model, background)
    
    # 4. 计算 SHAP 值
    shap_values = explainer.shap_values(test_samples)
    
    # 5. 可视化
    # 解决中文显示问题（如果需要）
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False

    # 柱状图：展示全局特征重要性
    plt.figure()
    shap.summary_plot(shap_values, test_samples.cpu().numpy(), feature_names=FEATURE_COLS, plot_type="bar", show=False)
    plt.title("DPG-Net 特征全局重要性分析 (真实数据)")
    plt.savefig("shap_bar_real.png", dpi=300, bbox_inches='tight')
    print("[+] 全局重要性柱状图已保存: shap_bar_real.png")

    # 蜂巢图：展示特征取值对预测的正负向影响
    plt.figure()
    shap.summary_plot(shap_values, test_samples.cpu().numpy(), feature_names=FEATURE_COLS, show=False)
    plt.title("DPG-Net 特征决策机制分析 (Beeswarm)")
    plt.savefig("shap_beeswarm_real.png", dpi=300, bbox_inches='tight')
    print("[+] 决策机制蜂巢图已保存: shap_beeswarm_real.png")

def evaluate_model(model, test_loader):
    print("\n[*] 正在评估常规测试集...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs, training_mode=False)
            predicted = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print(classification_report(y_true, y_pred, target_names=['Flash Event', 'DDoS Attack'], digits=4))

if __name__ == "__main__":
    # 训练并返回模型和测试数据
    trained_model, test_data_loader, X_test_raw = train_model()
    
    # 评估准确率
    evaluate_model(trained_model, test_data_loader)
    
    # 【核心】运行 SHAP 分析
    run_shap_analysis(trained_model, X_test_raw)