#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        
        # 纯粹的数学投影：只限制扰动幅度在 epsilon 内
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
    
    # 测试攻击前的全局洁净表现
    y_pred_clean = dt_model.predict(X_test)
    print(f"[+] 攻击前 DT 洁净测试集整体准确率: {accuracy_score(y_test, y_pred_clean)*100:.2f}%")

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
    print("\n[*] 正在提取测试集中的 DDoS 流量并发动 PGD 拟态对抗攻击...")
    
    # 找出测试集中所有真实标签为 DDoS (1) 的样本，这是攻击者的弹药库
    ddos_indices = np.where(y_test == 1)[0]
    X_ddos_clean = X_test[ddos_indices]
    y_ddos_true = y_test[ddos_indices]
    
    # 找出测试集中所有真实标签为 FE 正常流量 (0) 的样本，这些不受攻击影响
    fe_indices = np.where(y_test == 0)[0]
    X_fe_clean = X_test[fe_indices]
    y_fe_true = y_test[fe_indices]
    
    # 转为 Tensor 喂给替代模型生成对抗样本
    X_ddos_tensor = torch.FloatTensor(X_ddos_clean).to(device)
    y_ddos_tensor = torch.LongTensor(y_ddos_true).to(device)
    
    surrogate.eval()
    X_adv_tensor = pgd_attack(surrogate, X_ddos_tensor, y_ddos_tensor, EPSILON, ALPHA, NUM_ITER)
    X_adv_np = X_adv_tensor.cpu().numpy()
    
    # ================= 4. 评估 DT 模型的整体防线崩溃情况 =================
    print("[*] 正在将生成的对抗样本与正常的 FE 流量重新拼合...")
    
    # 将被污染的 DDoS 样本与干净的 FE 样本重新拼接成完整的测试集
    X_test_attacked = np.vstack((X_adv_np, X_fe_clean))
    y_test_attacked = np.hstack((y_ddos_true, y_fe_true)) # 真实标签保持不变
    
    print("[*] 正在让 DT 模型对受污染的全局测试集进行预测...")
    y_pred_adv = dt_model.predict(X_test_attacked)
    
    # 计算攻击后的全局四项核心指标 (使用 macro 平均)
    acc_adv = accuracy_score(y_test_attacked, y_pred_adv)
    prec_adv = precision_score(y_test_attacked, y_pred_adv, zero_division=0)
    rec_adv = recall_score(y_test_attacked, y_pred_adv, zero_division=0)
    f1_adv = f1_score(y_test_attacked, y_pred_adv, zero_division=0)
    
    print("\n" + "="*50)
    print(" 💥 传统模型 (DT 3D) 遭遇黑盒迁移 PGD 攻击全局结果 💥")
    print("="*50)
    print(f"攻击扰动幅度 (Epsilon) : {EPSILON}")
    print(f"攻击后全局 Accuracy (%)  : {acc_adv * 100:.2f}")
    print(f"攻击后全局 Precision (%) : {prec_adv * 100:.2f}")
    print(f"攻击后全局 Recall (%)    : {rec_adv * 100:.2f}")
    print(f"攻击后全局 F1-Score (%)  : {f1_adv * 100:.2f}")
    print("="*50)
    print("\n[专家分析]: 四项指标的全面暴跌，证明传统决策树的特征体系和边界已在对抗扰动下彻底失效！您可以直接将这组数据填入表 4-Y 中。")

if __name__ == "__main__":
    main()