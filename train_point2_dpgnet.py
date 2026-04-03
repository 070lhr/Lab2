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

# ================= 新增：PGD 对抗攻击生成模块 =================
def pgd_attack(model, X, y, epsilon=0.8, alpha=0.1, num_iter=10):
    """
    投影梯度下降 (PGD) 攻击
    严格对应大论文 4.3.5 节面向白盒攻击的鲁棒性基准公式 (4-21)
    """
    print(f"\n>>> [对抗攻击] 正在使用 PGD 算法生成对抗测试集 (epsilon={epsilon})...")
    # 临时禁用 cuDNN，以支持对 GRU 的输入求导
    torch.backends.cudnn.enabled = False
    model.eval()
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    criterion = nn.BCELoss()

    # 1. 随机初始化扰动 (在 -epsilon 到 epsilon 之间)
    delta = torch.empty_like(X_tensor).uniform_(-epsilon, epsilon)
    X_adv = X_tensor + delta

    # 记录数据集边界，防止扰动后超出物理实际意义的极值
    X_min, X_max = X_tensor.min(), X_tensor.max()
    X_adv = torch.clamp(X_adv, min=X_min, max=X_max)

    # 2. 多步迭代寻优
    for i in range(num_iter):
        X_adv.requires_grad = True
        outputs = model(X_adv)
        
        # 我们希望最大化损失函数，让模型分类错误
        loss = criterion(outputs, y_tensor)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            # 沿着梯度符号方向前进 (FGSM 的迭代版)
            adv_step = alpha * X_adv.grad.sign()
            X_adv = X_adv + adv_step
            
            # 投影回 L_inf 范数的 epsilon 邻域内
            eta = torch.clamp(X_adv - X_tensor, min=-epsilon, max=epsilon)
            X_adv = X_tensor + eta
            
            # 再次截断至特征真实边界
            X_adv = torch.clamp(X_adv, min=X_min, max=X_max)

    torch.backends.cudnn.enabled = True
    return X_adv.cpu().numpy()

def run_adversarial_shap_analysis(model, X_test_clean, y_test):
    print("\n>>> [阶段 4/4] 极限对抗环境下的 SHAP 归因分析...")
    
    # 1. 生成被高强度干扰的对抗测试集 (设定论文里的极限扰动 0.8)
    X_test_adv = pgd_attack(model, X_test_clean, y_test, epsilon=0.8, alpha=0.1, num_iter=10)
    
    # 2. 评估对抗攻击后的模型性能 (验证内生鲁棒性下界)
    model.eval()
    with torch.no_grad():
        outputs_adv = model(torch.tensor(X_test_adv, dtype=torch.float32).to(DEVICE))
        y_pred_adv = (outputs_adv > 0.5).cpu().numpy()
        
    print("\n>>> [对抗攻击后] 模型检测性能：")
    print("-" * 30)
    print(classification_report(y_test, y_pred_adv, target_names=['Flash', 'DDoS'], digits=4))
    print("-" * 30)

    # 3. 开始对抗样本的 SHAP 计算
    print("  [提示] 正在使用 GradientExplainer 解释【对抗样本】，耗时较长，请稍候...")
    torch.backends.cudnn.enabled = False
    
    # 背景样本依旧使用干净的，但解释的对象变成对抗样本
    bg_data = torch.tensor(X_test_clean[np.random.choice(X_test_clean.shape[0], 100, replace=False)], dtype=torch.float32).to(DEVICE)
    # 取 200 个对抗样本进行解释
    test_data_adv = torch.tensor(X_test_adv[np.random.choice(X_test_adv.shape[0], 200, replace=False)], dtype=torch.float32).to(DEVICE)
    
    start_time = time.time()
    explainer = shap.GradientExplainer(model, bg_data)
    shap_values = explainer.shap_values(test_data_adv)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    end_time = time.time()
    torch.backends.cudnn.enabled = True
    print(f"  [+] SHAP 计算完成，耗时: {end_time - start_time:.2f}s")

    # 配置字体 (保持原样)
    try:
        fm.fontManager.addfont(SIMSUN_FONT_PATH)
        fm.fontManager.addfont(TIMES_FONT_PATH)
        plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=TIMES_FONT_PATH).get_name(), 
                                           fm.FontProperties(fname=SIMSUN_FONT_PATH).get_name()]
        plt.rcParams['axes.unicode_minus'] = False 
    except Exception:
        plt.rcParams['axes.unicode_minus'] = False
    
    # 绘图并保存 (加上 adv 后缀加以区分)
    print("  [*] 正在生成对抗环境下的重要性柱状图...")
    plt.figure()
    shap.summary_plot(shap_values, test_data_adv.cpu().numpy(), feature_names=FEATURE_COLS, plot_type="bar", show=False)
    plt.savefig("shap_bar_adv.png", dpi=300, bbox_inches='tight')
    
    print("  [*] 正在生成对抗环境下的决策机制蜂群图...")
    plt.figure()
    shap.summary_plot(shap_values, test_data_adv.cpu().numpy(), feature_names=FEATURE_COLS, show=False)
    plt.savefig("shap_beeswarm_adv.png", dpi=300, bbox_inches='tight')
    
    print(f"\n[任务完成] 对抗攻击与解释结果已保存！")

# ================= 替换 Main 函数的执行流程 =================
if __name__ == "__main__":
    trained_model, test_loader, X_test_clean = train_model()
    evaluate_model(trained_model, test_loader)
    
    # 获取完整的 y_test 用于生成对抗样本
    y_test_full = torch.cat([y for _, y in test_loader], dim=0).numpy().squeeze()
    
    # 运行对抗攻击下的 SHAP 分析
    run_adversarial_shap_analysis(trained_model, X_test_clean, y_test_full)