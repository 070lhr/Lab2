import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm 
from tqdm import tqdm 

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
FLASH_FILE = './flash_event_9dim_full.csv'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'

# 如果没有这两个字体文件，可以注释掉下面绘图部分的字体设置，使用系统默认
SIMSUN_FONT_PATH = './SIMSUN.TTC'  
TIMES_FONT_PATH = './TIMES.TTF'    

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 50  # 如果想快速看个趋势，可以先改成 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    'Rate', 'Rate_Accel', 'Rate_CV',           # 前3维：动力学流 
    'SIP_Ent', 'SIPEnt_Change', 'SIPEnt_MA',   # 后6维：分布流
    'Size_Std', 'SizeStd_Change', 'SizeStd_MA' 
]
# ===========================================

class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

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

class DynBranchGRU(nn.Module):
    def __init__(self, in_features=3, hidden_dim=16):
        super(DynBranchGRU, self).__init__()
        self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1) 
        output, hidden = self.gru(x)
        return hidden.squeeze(0)

# ================= 核心修改：支持两路独立 Dropout 的网络 =================
class DPG_Net_Ablation(nn.Module):
    def __init__(self, dyn_drop_p=0.0, dist_drop_p=0.0): 
        super(DPG_Net_Ablation, self).__init__()
        self.dynamics_branch = DynBranchGRU(in_features=3, hidden_dim=16)
        self.dist_branch = DistBranchDCN(in_features=6, embed_dim=32, num_layers=2)
        
        # 独立控制两路特征的失活率
        self.dyn_drop_p = dyn_drop_p
        self.dist_drop_p = dist_drop_p
        
        self.classifier = nn.Linear(48, 1) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_dyn = self.dynamics_branch(x[:, 0:3])
        out_dist = self.dist_branch(x[:, 3:9])
        
        if self.training:
            # 动力学分支失活
            if self.dyn_drop_p > 0:
                mask_dyn = (torch.rand(out_dyn.size(0), 1, device=out_dyn.device) > self.dyn_drop_p).float()
                out_dyn_dropped = out_dyn * mask_dyn / (1.0 - self.dyn_drop_p)
            else:
                out_dyn_dropped = out_dyn
                
            # 分布分支失活
            if self.dist_drop_p > 0:
                mask_dist = (torch.rand(out_dist.size(0), 1, device=out_dist.device) > self.dist_drop_p).float()
                out_dist_dropped = out_dist * mask_dist / (1.0 - self.dist_drop_p)
            else:
                out_dist_dropped = out_dist
        else:
            out_dyn_dropped = out_dyn
            out_dist_dropped = out_dist
            
        h_concat = torch.cat((out_dyn_dropped, out_dist_dropped), dim=1)
        out = self.sigmoid(self.classifier(h_concat))
        return out

# ================= PGD 攻击 (维持原有物理约束) =================
def pgd_attack(model, X, y, epsilon, alpha=0.2, num_iter=20):
    if epsilon == 0.0:
        return X
    
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
            adv_step[:, 0:3] = 0  # 物理约束：不动前3维动力学特征
            
            X_adv = X_adv + adv_step
            eta = torch.clamp(X_adv - X_tensor, min=-epsilon, max=epsilon)
            X_adv = X_tensor + eta
            X_adv = torch.clamp(X_adv, min=X_min, max=X_max)

    torch.backends.cudnn.enabled = True
    return X_adv.cpu().numpy()

# ================= 模型训练与评估的自动化封装 =================
def train_and_evaluate(dyn_drop, dist_drop, X_train, y_train, X_test, y_test, epsilons):
    model = DPG_Net_Ablation(dyn_drop_p=dyn_drop, dist_drop_p=dist_drop).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCELoss()
    train_loader = DataLoader(TrafficDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"\n[*] 正在训练模型 [动力学 Drop={dyn_drop}, 分布 Drop={dist_drop}] ...")
    model.train()
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            
    # 开始评估不同 epsilon 下的表现
    results = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    model.eval()
    
    for eps in epsilons:
        X_test_adv = pgd_attack(model, X_test, y_test, epsilon=eps)
        with torch.no_grad():
            outputs_adv = model(torch.tensor(X_test_adv, dtype=torch.float32).to(DEVICE))
            y_pred = (outputs_adv > 0.5).cpu().numpy()
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results['acc'].append(acc)
        results['prec'].append(prec)
        results['rec'].append(rec)
        results['f1'].append(f1)
        
    return results

# ================= 绘图函数 =================
def plot_ablation_results(epsilons, results_dict):
    try:
        fm.fontManager.addfont(SIMSUN_FONT_PATH)
        fm.fontManager.addfont(TIMES_FONT_PATH)
        simsun_name = fm.FontProperties(fname=SIMSUN_FONT_PATH).get_name()
        times_name = fm.FontProperties(fname=TIMES_FONT_PATH).get_name()
        plt.rcParams['font.sans-serif'] = [times_name, simsun_name]
    except Exception:
        plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei'] 
        
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    styles = {
        'No_Drop': {'marker':'o', 'ls':'--', 'color':'gray', 'label':'无失活机制 (No Dropout)'},
        'Sym_Drop': {'marker':'s', 'ls':'-.', 'color':'#4A90E2', 'label':'对称失活机制 (Symmetric)'},
        'Asym_Drop': {'marker':'^', 'ls':'-', 'color':'#D0021B', 'lw':2, 'label':'非对称失活机制 (DPDADFE)'}
    }

    metrics = [
        ('acc', axs[0, 0], '准确率', '(a) 准确率'),
        ('prec', axs[0, 1], '精确率', '(b) 精确率'),
        ('rec', axs[1, 0], '召回率', '(c) 召回率'),
        ('f1', axs[1, 1], 'F1分数', '(d) F1分数')
    ]

    for key, ax, ylabel, title in metrics:
        ax.plot(epsilons, results_dict['No_Drop'][key], **styles['No_Drop'])
        ax.plot(epsilons, results_dict['Sym_Drop'][key], **styles['Sym_Drop'])
        ax.plot(epsilons, results_dict['Asym_Drop'][key], **styles['Asym_Drop'])
        
        ax.set_xlabel(r'对抗扰动半径 $\epsilon$', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, y=-0.2, fontsize=14) 
        ax.set_ylim(0.2, 1.05)
        ax.set_xticks(epsilons)
        ax.grid(True, linestyle='--', alpha=0.6)

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.52), 
               ncol=1, fontsize=12, framealpha=1.0, edgecolor='black')

    plt.subplots_adjust(hspace=0.35, wspace=0.2)
    plt.savefig('real_ablation_dropout.png', dpi=600, bbox_inches='tight')
    print("\n[+] 真实消融实验运行完毕，图表已保存为 'real_ablation_dropout.png'！")

# ================= 主控制流程 =================
def main():
    print(">>> 正在加载并预处理数据...")
    df_flash = pd.read_csv(FLASH_FILE)
    df_ddos = pd.read_csv(DDOS_FILE)
    X = np.concatenate([df_flash[FEATURE_COLS].values, df_ddos[FEATURE_COLS].values], axis=0)
    y = np.concatenate([np.zeros(len(df_flash)), np.ones(len(df_ddos))], axis=0)
    
    # 抽取子集测试可以加快实验速度，这里默认使用全集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    epsilons = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    # 实验配置定义 (dyn_drop, dist_drop)
    configs = {
        'No_Drop': (0.0, 0.0),      # 两路都不丢弃
        'Sym_Drop': (0.5, 0.5),     # 两路丢弃率相同
        'Asym_Drop': (0.3, 0.8)     # 动力学丢弃低，分布丢弃高 (你的 DPDADFE 架构)
    }
    
    all_results = {}
    for name, (dyn_p, dist_p) in configs.items():
        res = train_and_evaluate(dyn_p, dist_p, X_train, y_train, X_test, y_test, epsilons)
        all_results[name] = res
        
    print("\n>>> 开始绘制对比结果图表...")
    plot_ablation_results(epsilons, all_results)

if __name__ == "__main__":
    main()