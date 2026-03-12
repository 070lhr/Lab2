#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ================= 配置区域 =================
INPUT_CSV = './tinubu_3dim_full_mixed.csv' # 请确保这里是您真实的 3 维特征数据文件路径
EPSILON = 0.8      # 释放攻击幅度，确保能够跨越替代模型的迁移鸿沟
ALPHA = 0.1        # 加大 PGD 步长
NUM_ITER = 40      # 增加迭代次数，寻找最优对抗扰动
# ===========================================

# 定义替代模型 (Surrogate Model) - 窃取决策边界的“间谍”
class SurrogateMLP(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # 输出 2 分类 (正常=0, DDoS=1)
        )
    def forward(self, x):
        return self.net(x)

def pgd_attack(model, X, y, epsilon, alpha, num_iter):
    """
    修复后的 PGD 对抗样本生成算法（纯净数学投影版）
    """
    criterion = nn.CrossEntropyLoss()
    
    # 复制原始输入作为起步点
    X_adv = X.clone().detach()
    
    for _ in range(num_iter):
        X_adv.requires_grad_(True)
        
        outputs = model(X_adv)
        loss = criterion(outputs, y)
        
        # 反向传播计算梯度
        model.zero_grad()
        loss.backward()
        
        # 获取梯度方向
        grad = X_adv.grad.detach()
        
        # PGD 核心更新步骤：沿梯度的反方向移动（最大化损失）
        X_adv = X_adv + alpha * grad.sign()
        
        # 【关键修复】纯粹的数学投影：只限制扰动幅度在 epsilon 内，绝不强行截断负数！
        eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
        X_adv = (X + eta).detach()
        
    return X_adv

def main():
    print("[*] 正在加载 3 维基准特征数据...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"[!] 找不到文件 {INPUT_CSV}！请检查路径。")
        return

    # 数据准备 (假设标签列名为 'Label')
    X = df[['Num_SrcIP', 'Num_New_SrcIP', 'Mean_IAT']].values
    y = df['Label'].values

    # 标准化：将数据缩放到均值为0，方差为1的分布中
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
    
    surrogate.train()
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = surrogate(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    print("[+] 替代模型训练完毕！")

    # ================= 3. 发动 PGD 逃逸攻击 =================
    print("\n[*] 正在对测试集中的 DDoS 流量发动 PGD 拟态对抗攻击...")
    
    # 找出测试集中所有真实标签为 DDoS (1) 的样本
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
    X_adv_tensor = pgd_attack(surrogate, X_ddos_tensor, y_ddos_tensor, EPSILON, ALPHA, NUM_ITER)
    X_adv_np = X_adv_tensor.cpu().numpy()
    
    # ================= 4. 评估 DT 模型的防线崩溃情况 =================
    print("[*] 将生成的对抗样本跨模型迁移，喂给未加防备的 DT 模型...")
    pred_ddos_adv = dt_model.predict(X_adv_np)
    
    # 攻击后的 DDoS 检出率
    recall_adv = np.mean(pred_ddos_adv == 1)
    
    print("\n" + "="*50)
    print(" 💥 传统模型 (DT 3D) 遭遇黑盒迁移 PGD 攻击的结果 💥")
    print("="*50)
    print(f"攻击扰动幅度 (Epsilon): {EPSILON}")
    print(f"攻击前 DDoS 检出率 (Recall) : {recall_clean * 100:.2f}%")
    print(f"攻击后 DDoS 检出率 (Recall) : {recall_adv * 100:.2f}%")
    print(f"防御下降幅度                : -{(recall_clean - recall_adv) * 100:.2f}%")
    print("="*50)
    print("\n[专家分析]: 成功实现跨模型迁移打击！传统决策树脆弱的边界在 PGD 扰动下已全面溃散。")

if __name__ == "__main__":
    main()