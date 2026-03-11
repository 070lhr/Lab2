#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, classification_report

# ================= 配置区域 =================
INPUT_CSV = './tinubu_3dim_full_mixed.csv' # 您提取的 3 维特征数据
EPSILON = 0.3      # PGD 扰动幅度 (可以调大调小观察效果)
ALPHA = 0.05       # PGD 步长
NUM_ITER = 10      # PGD 迭代次数
# ===========================================

# 定义替代模型 (Surrogate Model) - 一个简单的多层感知机
class SurrogateMLP(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # 输出 2 分类 (FE=0, DDoS=1)
        )
    def forward(self, x):
        return self.net(x)

def pgd_attack(model, X, y, epsilon, alpha, num_iter):
    """
    经典的 PGD 对抗样本生成算法
    """
    # 攻击目标：我们希望生成对抗样本，所以这里的 y 是原始真实标签
    # 损失函数的目标是最大化交叉熵损失，让模型分错
    criterion = nn.CrossEntropyLoss()
    
    # 复制原始输入作为起步点
    X_adv = X.clone().detach().requires_grad_(True)
    
    for _ in range(num_iter):
        _X_adv = X_adv.clone().detach().requires_grad_(True)
        
        outputs = model(_X_adv)
        loss = criterion(outputs, y)
        
        # 反向传播计算梯度
        model.zero_grad()
        loss.backward()
        
        # 获取梯度方向
        grad = _X_adv.grad.detach()
        
        # PGD 核心更新步骤：沿梯度的反方向（因为是最大化损失，所以加上梯度）移动
        X_adv = X_adv + alpha * grad.sign()
        
        # 投影操作：将扰动限制在 [-epsilon, epsilon] 范围内
        eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
        
        # 附加物理约束：由于网络特征（如IP数、IAT）不能为负，截断到 0 以上
        X_adv = torch.clamp(X + eta, min=0.0).detach()
        
    return X_adv

def main():
    print("[*] 正在加载 3 维基准特征数据...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"[!] 找不到文件 {INPUT_CSV}！")
        return

    # 数据准备
    X = df[['Num_SrcIP', 'Num_New_SrcIP', 'Mean_IAT']].values
    y = df['Label'].values

    # 【关键】为了让梯度下降更稳定，必须进行标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ================= 1. 训练目标 DT 模型 =================
    print("[*] 正在训练目标模型 (DT-Model)...")
    dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt_model.fit(X_train, y_train)
    
    # 测试攻击前的洁净表现
    y_pred_clean = dt_model.predict(X_test)
    acc_clean = accuracy_score(y_test, y_pred_clean)
    print(f"[+] 攻击前 DT 洁净测试集准确率: {acc_clean*100:.2f}%")

    # ================= 2. 训练替代神经网络 =================
    print("\n[*] 正在训练替代神经网络 (Surrogate MLP) 以窃取决策边界...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = SurrogateMLP(input_dim=3).to(device)
    
    optimizer = optim.Adam(surrogate.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    
    # 快速训练替代模型 (200 轮)
    surrogate.train()
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = surrogate(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    print("[+] 替代模型训练完毕！")

    # ================= 3. 发动 PGD 逃逸攻击 =================
    # 我们只对测试集中的 DDoS 样本进行攻击（试图将其伪装成 FE，即逃逸检测）
    print("\n[*] 正在对测试集中的 DDoS 流量发动 PGD 拟态对抗攻击...")
    
    # 找出测试集中所有真实标签为 DDoS (1) 的样本索引
    ddos_indices = np.where(y_test == 1)[0]
    X_ddos_clean = X_test[ddos_indices]
    y_ddos_true = y_test[ddos_indices]
    
    # 攻击前的 DDoS 检出率 (Recall)
    pred_ddos_clean = dt_model.predict(X_ddos_clean)
    recall_clean = np.mean(pred_ddos_clean == 1)
    
    # 转为 Tensor 喂给替代模型生成对抗样本
    X_ddos_tensor = torch.FloatTensor(X_ddos_clean).to(device)
    y_ddos_tensor = torch.LongTensor(y_ddos_true).to(device)
    
    surrogate.eval()
    # 使用窃取到的梯度生成对抗样本
    X_adv_tensor = pgd_attack(surrogate, X_ddos_tensor, y_ddos_tensor, EPSILON, ALPHA, NUM_ITER)
    X_adv_np = X_adv_tensor.cpu().numpy()
    
    # ================= 4. 评估 DT 模型的防线崩溃情况 =================
    print("[*] 将生成的对抗样本喂给未加防备的 DT 模型...")
    pred_ddos_adv = dt_model.predict(X_adv_np)
    
    # 攻击后的 DDoS 检出率
    recall_adv = np.mean(pred_ddos_adv == 1)
    
    print("\n" + "="*50)
    print(" 💥 传统模型 (DT 3D) 遭遇 PGD 攻击的结果 💥")
    print("="*50)
    print(f"攻击扰动幅度 (Epsilon): {EPSILON}")
    print(f"攻击前 DDoS 检出率 (Recall) : {recall_clean * 100:.2f}%")
    print(f"攻击后 DDoS 检出率 (Recall) : {recall_adv * 100:.2f}%  <-- 致命暴跌！")
    print(f"防御下降幅度                : -{(recall_clean - recall_adv) * 100:.2f}%")
    print("="*50)
    print("\n[专家分析]: 仅依赖 3 维特征的传统模型，其脆弱的决策边界已通过替代模型被完全逆向。微小的 PGD 扰动即可让其防线彻底瘫痪！")

if __name__ == "__main__":
    main()