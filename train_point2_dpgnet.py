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

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
FLASH_FILE = './flash_event_9dim_full.csv'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
MODEL_SAVE_PATH = './dpg_net_model.pth'
SCALER_SAVE_PATH = './dpg_scaler.pkl'

# 本地字体文件路径配置 (请确保文件名大小写和实际文件一致)
SIMSUN_FONT_PATH = './SIMSUN.TTC'  # 宋体文件路径
TIMES_FONT_PATH = './TIMES.TTF'    # Times New Roman 文件路径

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 50
DROPOUT_RATE = 0.3  # 融合层的隐式特征丢弃率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 严格匹配定义的 9 维双流特征列名
FEATURE_COLS = [
    'Rate', 'Rate_Accel', 'Rate_CV',           # 动力学流 
    'SIP_Ent', 'SIPEnt_Change', 'SIPEnt_MA',   # 分布流-熵 
    'Size_Std', 'SizeStd_Change', 'SizeStd_MA' # 分布流-载荷 
]
# ===========================================

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ================= 1. 统计分布特征：显式交叉网络 (DCN) =================
class CrossLayer(nn.Module):
    """
    单层显式交叉网络：
    x_{l} = x_{0} ⊙ (x_{l-1} W_{cross}^l + b_{cross}^l) + x_{l-1}
    """
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
    """
    基于显式交叉机制的统计分布特征映射支路
    输入6维 -> 扩展至32维 -> 2层交叉网络
    """
    def __init__(self, in_features=6, embed_dim=32, num_layers=2):
        super(DistBranchDCN, self).__init__()
        self.expand_layer = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.ReLU()
        )
        self.cross_layers = nn.ModuleList([CrossLayer(embed_dim) for _ in range(num_layers)])

    def forward(self, x):
        x0 = self.expand_layer(x)  
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)     
        return xl                  

# ================= 2. 动力学特征：门控循环单元 (GRU) =================
class DynBranchGRU(nn.Module):
    """
    基于门控循环单元的动力学特征时序聚合支路
    输入3维 -> 输出16维
    """
    def __init__(self, in_features=3, hidden_dim=16):
        super(DynBranchGRU, self).__init__()
        self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        # 增加 seq_len=1 维度以适配 GRU 的三维输入要求 (batch, seq, feature)
        x = x.unsqueeze(1) 
        output, hidden = self.gru(x)
        return hidden.squeeze(0)

# ================= 3. 融合网络：DPDADFE 核心架构 =================
class DPG_Net(nn.Module):
    """
    双流感知应用层 DDoS 攻击与 FE 区分模型总体架构
    """
    def __init__(self, dropout_rate=0.3):
        super(DPG_Net, self).__init__()
        self.dynamics_branch = DynBranchGRU(in_features=3, hidden_dim=16)
        self.dist_branch = DistBranchDCN(in_features=6, embed_dim=32, num_layers=2)
        
        # 隐式特征丢弃正则化机制
        self.fusion_dropout = nn.Dropout(p=dropout_rate)
        
        # 全局融合决策层，输入维度 32 + 16 = 48
        self.classifier = nn.Linear(48, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 异构特征分离提取
        out_dyn = self.dynamics_branch(x[:, 0:3])
        out_dist = self.dist_branch(x[:, 3:9])
        
        # 2. 特征级联：构建 48 维全局联合表征向量
        h_concat = torch.cat((out_dyn, out_dist), dim=1)
        
        # 3. 隐式特征丢弃机制与分类输出
        h_drop = self.fusion_dropout(h_concat)
        out = self.sigmoid(self.classifier(h_drop))
        
        return out

# ================= 主流程函数 =================
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
    model = DPG_Net(dropout_rate=DROPOUT_RATE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    pbar = tqdm(range(EPOCHS), desc="训练进度")
    for epoch in pbar:
        model.train()  # 开启训练模式，激活 Dropout
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
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
    model.eval()  # 开启评估模式，关闭 Dropout
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(DEVICE))
            y_true.extend(labels.numpy())
            y_pred.extend((outputs > 0.5).cpu().numpy())
    
    print("-" * 30)
    print(classification_report(y_true, y_pred, target_names=['Flash', 'DDoS'], digits=4))
    print("-" * 30)

def run_shap_analysis(model, X_test):
    print("\n>>> [阶段 4/4] SHAP 解释性归因分析...")
    print("  [提示] 正在使用 GradientExplainer 计算高维特征梯度，这可能需要 1-3 分钟，请稍候...")
    
    model.eval()  
    # 继续保持禁用 cuDNN 加速，以支持 GRU 的反向传播图提取
    torch.backends.cudnn.enabled = False
    
    # 抽取背景样本和待解释样本
    bg_data = torch.tensor(X_test[np.random.choice(X_test.shape[0], 100, replace=False)], dtype=torch.float32).to(DEVICE)
    test_data = torch.tensor(X_test[np.random.choice(X_test.shape[0], 200, replace=False)], dtype=torch.float32).to(DEVICE)
    
    start_time = time.time()
    
    # ================= 核心修复 =================
    # 弃用 DeepExplainer，改用 GradientExplainer
    # 完美解决由于 DCN 显式交叉层张量乘法 (x0 * cross) 导致的加和属性(Additivity)崩溃问题
    explainer = shap.GradientExplainer(model, bg_data)
    shap_values = explainer.shap_values(test_data)
    
    # 针对 PyTorch 单节点输出，GradientExplainer 可能返回包含单元素的 list，此处做解包处理
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    # ============================================
    
    end_time = time.time()
    
    # 恢复环境配置
    torch.backends.cudnn.enabled = True
    
    print(f"  [+] SHAP 计算完成，耗时: {end_time - start_time:.2f}s")

    print("  [*] 正在配置本地双字体 (Times New Roman + SimSun)...")
    try:
        fm.fontManager.addfont(SIMSUN_FONT_PATH)
        fm.fontManager.addfont(TIMES_FONT_PATH)
        
        simsun_name = fm.FontProperties(fname=SIMSUN_FONT_PATH).get_name()
        times_name = fm.FontProperties(fname=TIMES_FONT_PATH).get_name()
        
        plt.rcParams['font.sans-serif'] = [times_name, simsun_name]
        plt.rcParams['axes.unicode_minus'] = False 
    except Exception as e:
        print(f"  [!] 字体加载失败，请检查文件路径是否正确。错误信息: {e}")
        print(f"  [!] 将使用系统默认字体回退方案...")
        plt.rcParams['axes.unicode_minus'] = False
    
    print("  [*] 正在生成特征重要性柱状图...")
    plt.figure()
    shap.summary_plot(shap_values, test_data.cpu().numpy(), feature_names=FEATURE_COLS, plot_type="bar", show=False)
    plt.savefig("shap_bar_real.png", dpi=300, bbox_inches='tight')
    
    print("  [*] 正在生成决策机制蜂群图...")
    plt.figure()
    shap.summary_plot(shap_values, test_data.cpu().numpy(), feature_names=FEATURE_COLS, show=False)
    plt.savefig("shap_beeswarm_real.png", dpi=300, bbox_inches='tight')
    
    print(f"\n[任务完成] 所有结果已保存至当前目录。")
    
if __name__ == "__main__":
    trained_model, test_loader, X_test_raw = train_model()
    evaluate_model(trained_model, test_loader)
    run_shap_analysis(trained_model, X_test_raw)