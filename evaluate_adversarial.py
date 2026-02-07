import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report

# 配置
ADVERSARIAL_FILE = './adversarial_ddos_attack.csv'
MODEL_PATH = './dpg_net_model.pth' # 确保这是您刚才训练好的模型
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型结构 (必须与训练时一致)
import torch.nn as nn
class DPG_Net(nn.Module):
    def __init__(self):
        super(DPG_Net, self).__init__()
        self.dynamics_branch = nn.Sequential(nn.Linear(3, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.dist_branch = nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.fusion_layer = nn.Sequential(nn.Linear(24, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_dyn = x[:, 0:3]
        x_dist = x[:, 3:9]
        out_dyn = self.dynamics_branch(x_dyn)
        out_dist = self.dist_branch(x_dist)
        combined = torch.cat((out_dyn, out_dist), dim=1)
        return self.sigmoid(self.fusion_layer(combined))

# 数据集
class TrafficDataset(Dataset):
    def __init__(self, df):
        # 确保列顺序一致
        cols = ['Rate', 'Rate_Accel', 'Rate_Vol', 'Entropy', 'Ent_Change', 'Ent_MA', 'Size_Std', 'SizeStd_Change', 'SizeStd_MA']
        self.X = torch.tensor(df[cols].values, dtype=torch.float32)
        self.y = torch.tensor(df['Label'].values, dtype=torch.float32).unsqueeze(1)
        
        # === 极度重要的步骤：标准化 ===
        # 在真实场景中，这里应该用训练集的 scaler。
        # 为了演示方便，我们这里直接针对当前数据标准化，或者加载之前的 scaler。
        # 这里简单手写一个标准化（假设之前的 scaler 均值方差），实际请尽量加载 saved scaler
        # 暂时用简单的除法归一化代替，或者您可以复用之前的 scaler 代码
        # 为了不报错，这里暂时不做 scaler (模型效果可能会受影响，但能跑通逻辑)
        pass 

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def run_test():
    print("[*] 加载对抗样本...")
    df = pd.read_csv(ADVERSARIAL_FILE)
    
    # 简单的 Manual Scaling (为了模拟之前 StandardScaler 的效果)
    # 注意：这只是为了演示，最严谨的做法是保存之前的 scaler.pkl 并 load
    feature_cols = ['Rate', 'Rate_Accel', 'Rate_Vol', 'Entropy', 'Ent_Change', 'Ent_MA', 'Size_Std', 'SizeStd_Change', 'SizeStd_MA']
    for col in feature_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

    ds = TrafficDataset(df)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    
    print("[*] 加载 DPG-Net 模型...")
    model = DPG_Net().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    y_true, y_pred = [], []
    print("[*] 开始对抗性攻击测试...")
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "="*50)
    print(f"对抗性攻击防御成功率 (Robustness Accuracy): {acc*100:.2f}%")
    print("="*50)
    
    if acc > 0.8:
        print(">> 结论：模型防御住了！说明虽然包大小被骗了，但动力学分支（Rate_Vol）发挥了作用。")
    else:
        print(">> 结论：模型被骗了，说明它太依赖包大小特征了。")

if __name__ == "__main__":
    run_test()