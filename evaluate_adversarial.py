import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import joblib # 需要加载 scaler

# ================= 配置 =================
ADVERSARIAL_FILE = './adversarial_ddos_attack.csv' # 之前生成的纯欺骗样本
MODEL_PATH = './dpg_net_model.pth'
SCALER_PATH = './dpg_scaler.pkl' # 刚才训练保存的
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================

# 模型定义 (必须一致)
class DPG_Net(nn.Module):
    def __init__(self):
        super(DPG_Net, self).__init__()
        self.dynamics_branch = nn.Sequential(nn.Linear(3, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.dist_branch = nn.Sequential(nn.Linear(6, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.fusion_layer = nn.Sequential(nn.Linear(16 + 32, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, training_mode=False):
        x_dyn = x[:, 0:3]
        x_dist = x[:, 3:9]
        out_dyn = self.dynamics_branch(x_dyn)
        out_dist = self.dist_branch(x_dist)
        combined = torch.cat((out_dyn, out_dist), dim=1)
        return self.sigmoid(self.fusion_layer(combined))

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def run_final_check():
    print("[*] 正在加载对抗样本 (Size_Std 被篡改为 Flash)...")
    df = pd.read_csv(ADVERSARIAL_FILE)
    
    feature_cols = ['Rate', 'Rate_Accel', 'Rate_Vol', 'Entropy', 'Ent_Change', 'Ent_MA', 'Size_Std', 'SizeStd_Change', 'SizeStd_MA']
    X = df[feature_cols].values
    y = df['Label'].values # 应该是全 1
    
    print("[*] 加载标准化器 (Scaler)...")
    scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(X) # 必须用训练时的参数来缩放
    
    ds = TrafficDataset(X, y)
    loader = DataLoader(ds, batch_size=64)
    
    print("[*] 加载模型...")
    model = DPG_Net().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    y_true, y_pred = [], []
    print("[*] 开始终极测试...")
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs, training_mode=False)
            preds = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*50)
    print(f"对抗性攻击防御成功率: {acc*100:.2f}%")
    print("="*50)
    
    if acc > 0.9:
        print(">> 完美！模型利用 Rate_Vol 识破了伪装！")
    elif acc > 0.8:
        print(">> 优秀！大部分攻击被拦截。")
    else:
        print(">> 依然有漏洞，可能需要加大 Feature Dropout 的概率。")

if __name__ == "__main__":
    run_final_check()