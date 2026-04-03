import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib 
import os
import shap 
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  
from tqdm import tqdm  
import time

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
FLASH_FILE = './flash_event_9dim_full.csv'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
MODEL_SAVE_PATH = './dpg_net_model.pth'
SCALER_SAVE_PATH = './dpg_scaler.pkl'

SIMSUN_FONT_PATH = './SIMSUN.TTC'  
TIMES_FONT_PATH = './TIMES.TTF'    

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    'Rate', 'Rate_Accel', 'Rate_CV',           # 动力学流 
    'SIP_Ent', 'SIPEnt_Change', 'SIPEnt_MA',   # 分布流
    'Size_Std', 'SizeStd_Change', 'SizeStd_MA' 
]
# ===========================================

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ================= 1. 统计分布特征网络 (DCN) =================
class CrossLayer(nn.Module):
    def __init__(self, dim):
        super(CrossLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        self.bias = nn.Parameter(torch.Tensor(dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        cross_term = torch.matmul(xl, self.weight) + self.bias
        return x0 * cross_term + xl

class DistBranchDCN(nn.Module):
    def __init__(self, in_features=6, embed_dim=32, num_layers=2):
        super(DistBranchDCN, self).__init__()
        self.expand_layer = nn.Sequential(nn.Linear(in_features, embed_dim), nn.ReLU())
        self.cross_layers = nn.ModuleList([CrossLayer(embed_dim) for _ in range(num_layers)])

    def forward(self, x):
        x0 = self.expand_layer(x)  
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)     
        return xl                  

# ================= 2. 动力学特征网络 (GRU) =================
class DynBranchGRU(nn.Module):
    def __init__(self, in_features=3, hidden_dim=16):
        super(DynBranchGRU, self).__init__()
        self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1) 
        output, hidden = self.gru(x)
        return hidden.squeeze(0)

# ================= 3. 核心架构：DPDADFE (回归论文纯粹数学设计) =================
class DPG_Net(nn.Module):
    def __init__(self, dropout_rate=0.3): # 回归 0.3 的温和丢弃率
        super(DPG_Net, self).__init__()
        self.dynamics_branch = DynBranchGRU(in_features=3, hidden_dim=16)
        self.dist_branch = DistBranchDCN(in_features=6, embed_dim=32, num_layers=2)
        
        # 严格对应公式 (4-18): 隐式特征丢弃正则化机制，作用于联合特征
        self.fusion_dropout = nn.Dropout(p=dropout_rate)
        
        # 严格对应公式 (4-19): 单层线性映射
        self.classifier = nn.Linear(48, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_dyn = self.dynamics_branch(x[:, 0:3])
        out_dist = self.dist_branch(x[:, 3:9])
        
        h_concat = torch.cat((out_dyn, out_dist), dim=1)
        h_drop = self.fusion_dropout(h_concat)
        out = self.sigmoid(self.classifier(h_drop))
        return out

# ================= PGD 攻击 (含物理约束修正) =================
def pgd_attack(model, X, y, epsilon=0.8, alpha=0.2, num_iter=20):
    print(f"\n>>> [对抗生成] 执行受物理约束的高强度 PGD 算法 (epsilon={epsilon})...")
    torch.backends.cudnn.enabled = False
    model.eval()
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    criterion = nn.BCELoss()

    delta = torch.zeros_like(X_tensor)
    delta[:, 3:9].uniform_(-epsilon, epsilon)
    X_adv = X_tensor + delta

    X_min, X_max = X_tensor.min(), X_tensor.max()
    X_adv = torch.clamp(X_adv, min=X_min, max=X_max)

    for i in range(num_iter):
        X_adv.requires_grad = True
        outputs = model(X_adv)
        loss = criterion(outputs, y_tensor)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            adv_step = alpha * X_adv.grad.sign()
            adv_step[:, 0:3] = 0 # 物理约束：动力学特征不被篡改
            X_adv = X_adv + adv_step
            eta = torch.clamp(X_adv - X_tensor, min=-epsilon, max=epsilon)
            X_adv = X_tensor + eta
            X_adv = torch.clamp(X_adv, min=X_min, max=X_max)

    torch.backends.cudnn.enabled = True
    return X_adv.cpu().numpy()

# ================= SHAP 可视化分析函数 (KernelExplainer) =================
def generate_shap_plots(model, X_test, bg_data, title_prefix, file_suffix):
    model.eval()
    print(f"  [*] 正在计算 {title_prefix} 环境的 SHAP 归因，请稍候...")
    start_time = time.time()
    
    def predict_fn(x_numpy):
        with torch.no_grad():
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(DEVICE)
            return model(x_tensor).cpu().numpy()
    
    # 聚类加速黑盒测算
    bg_summary = shap.kmeans(bg_data.cpu().numpy(), 15)
    explainer = shap.KernelExplainer(predict_fn, bg_summary)
    
    shap_values = explainer.shap_values(X_test.cpu().numpy(), silent=True)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    print(f"  [+] SHAP 计算完成，耗时: {time.time() - start_time:.2f}s")

    try:
        fm.fontManager.addfont(SIMSUN_FONT_PATH)
        fm.fontManager.addfont(TIMES_FONT_PATH)
        simsun_name = fm.FontProperties(fname=SIMSUN_FONT_PATH).get_name()
        times_name = fm.FontProperties(fname=TIMES_FONT_PATH).get_name()
        plt.rcParams['font.sans-serif'] = [times_name, simsun_name]
        plt.rcParams['axes.unicode_minus'] = False 
    except Exception:
        plt.rcParams['axes.unicode_minus'] = False

    plt.figure()
    shap.summary_plot(shap_values, X_test.cpu().numpy(), feature_names=FEATURE_COLS, plot_type="bar", show=False)
    plt.savefig(f"shap_bar_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values, X_test.cpu().numpy(), feature_names=FEATURE_COLS, show=False)
    plt.savefig(f"shap_beeswarm_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ================= 主控制流程 =================
def main():
    print("\n>>> [阶段 1/4] 数据预处理与标准化...")
    df_flash = pd.read_csv(FLASH_FILE)
    df_ddos = pd.read_csv(DDOS_FILE)
    X = np.concatenate([df_flash[FEATURE_COLS].values, df_ddos[FEATURE_COLS].values], axis=0)
    y = np.concatenate([np.zeros(len(df_flash)), np.ones(len(df_ddos))], axis=0)
    X_train, X_test_clean, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_clean = scaler.transform(X_test_clean)
    
    # 2. 模型训练
    print(f"\n>>> [阶段 2/4] DPG-Net 模型训练...")
    train_loader = DataLoader(TrafficDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    model = DPG_Net(dropout_rate=0.3).to(DEVICE) # 恢复 0.3
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    for epoch in tqdm(range(EPOCHS), desc="训练进度"):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
    
    # 3. 洁净环境性能评估与 SHAP
    print("\n>>> [阶段 3/4] 洁净环境：无扰动场景评估...")
    model.eval()
    with torch.no_grad():
        outputs_clean = model(torch.tensor(X_test_clean, dtype=torch.float32).to(DEVICE))
        y_pred_clean = (outputs_clean > 0.5).cpu().numpy()
    print(classification_report(y_test, y_pred_clean, target_names=['Flash', 'DDoS'], digits=4))
    
    # 【关键逻辑】：Clean 状态的解释，背景和测试集都是 Clean 数据
    bg_data_clean = torch.tensor(X_test_clean[np.random.choice(X_test_clean.shape[0], 100, replace=False)], dtype=torch.float32).to(DEVICE)
    test_data_clean = torch.tensor(X_test_clean[np.random.choice(X_test_clean.shape[0], 200, replace=False)], dtype=torch.float32).to(DEVICE)
    generate_shap_plots(model, test_data_clean, bg_data_clean, "洁净", "clean")

    # 4. 对抗环境性能评估与 SHAP
    print("\n>>> [阶段 4/4] 对抗环境：极限扰动场景评估...")
    X_test_adv = pgd_attack(model, X_test_clean, y_test, epsilon=0.8)
    with torch.no_grad():
        outputs_adv = model(torch.tensor(X_test_adv, dtype=torch.float32).to(DEVICE))
        y_pred_adv = (outputs_adv > 0.5).cpu().numpy()
    print(classification_report(y_test, y_pred_adv, target_names=['Flash', 'DDoS'], digits=4))
    
    # 【核心绝杀逻辑】：Adv 状态的解释，背景必须也是 Adv 数据！
    bg_data_adv = torch.tensor(X_test_adv[np.random.choice(X_test_adv.shape[0], 100, replace=False)], dtype=torch.float32).to(DEVICE)
    test_data_adv = torch.tensor(X_test_adv[np.random.choice(X_test_adv.shape[0], 200, replace=False)], dtype=torch.float32).to(DEVICE)
    generate_shap_plots(model, test_data_adv, bg_data_adv, "对抗", "adv")
    
    print("\n[任务完成] 祝你大论文图表绘制顺利！")

if __name__ == "__main__":
    main()