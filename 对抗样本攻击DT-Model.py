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
INPUT_CSV = './tinubu_3dim_full_mixed.csv' 
EPSILON = 0.8      
ALPHA = 0.1        
NUM_ITER = 40      
# ===========================================

class SurrogateMLP(nn.Module):
    def __init__(self, input_dim):
        super(SurrogateMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )
    def forward(self, x):
        return self.net(x)

def pgd_attack(model, X, y, epsilon, alpha, num_iter):
    criterion = nn.CrossEntropyLoss()
    X_adv = X.clone().detach()
    
    for _ in range(num_iter):
        X_adv.requires_grad_(True)
        
        outputs = model(X_adv)
        loss = criterion(outputs, y)
        
        model.zero_grad()
        loss.backward()
        
        grad = X_adv.grad.detach()
        X_adv = X_adv + alpha * grad.sign()
        
        eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
        X_adv = (X + eta).detach()
        
    return X_adv

def main():
    print("[*] 正在加载并重组 3 维基准特征数据...")
    try:
        df_raw = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"[!] 找不到文件 {INPUT_CSV}！请检查路径。")
        return

    # 分离正常流量(0)和攻击流量(1)
    df_fe = df_raw[df_raw['Label'] == 0]
    df_ddos = df_raw[df_raw['Label'] == 1]
    
    # 动态获取两者中数量较少的一方进行 1 比 1 完美平衡欠采样
    min_samples = min(len(df_fe), len(df_ddos))
    df_fe_sampled = df_fe.sample(n=min_samples, random_state=42)
    df_ddos_sampled = df_ddos.sample(n=min_samples, random_state=42)
    
    # 重新拼合并打乱顺序
    df = pd.concat([df_fe_sampled, df_ddos_sampled]).sample(frac=1.0, random_state=42)
    
    print("[+] 数据 1 比 1 平衡重组完毕！")
    print(f"    正常流量 (FE) 样本数 = {len(df_fe_sampled)}")
    print(f"    攻击流量 (DDoS) 样本数 = {len(df_ddos_sampled)}")

    X = df[['Num_SrcIP', 'Num_New_SrcIP', 'Mean_IAT']].values
    y = df['Label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[*] 正在训练目标模型 (DT-Model)...")
    dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt_model.fit(X_train, y_train)
    
    y_pred_clean = dt_model.predict(X_test)
    print(f"[+] 攻击前 DT 洁净测试集整体准确率 = {accuracy_score(y_test, y_pred_clean)*100:.2f}%")

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

    print("\n[*] 正在提取测试集中的 DDoS 流量并发动 PGD 拟态对抗攻击...")
    
    ddos_indices = np.where(y_test == 1)[0]
    X_ddos_clean = X_test[ddos_indices]
    y_ddos_true = y_test[ddos_indices]
    
    fe_indices = np.where(y_test == 0)[0]
    X_fe_clean = X_test[fe_indices]
    y_fe_true = y_test[fe_indices]
    
    X_ddos_tensor = torch.FloatTensor(X_ddos_clean).to(device)
    y_ddos_tensor = torch.LongTensor(y_ddos_true).to(device)
    
    surrogate.eval()
    X_adv_tensor = pgd_attack(surrogate, X_ddos_tensor, y_ddos_tensor, EPSILON, ALPHA, NUM_ITER)
    X_adv_np = X_adv_tensor.cpu().numpy()
    
    print("[*] 正在将生成的对抗样本与正常的 FE 流量重新拼合...")
    X_test_attacked = np.vstack((X_adv_np, X_fe_clean))
    y_test_attacked = np.hstack((y_ddos_true, y_fe_true)) 
    
    print("[*] 正在让 DT 模型对受污染的全局测试集进行预测...")
    y_pred_adv = dt_model.predict(X_test_attacked)
    
    acc_adv = accuracy_score(y_test_attacked, y_pred_adv)
    prec_adv = precision_score(y_test_attacked, y_pred_adv, zero_division=0)
    rec_adv = recall_score(y_test_attacked, y_pred_adv, zero_division=0)
    f1_adv = f1_score(y_test_attacked, y_pred_adv, zero_division=0)
    
    print("\n" + "="*50)
    print(" 传统模型 (DT 3D) 遭遇黑盒迁移 PGD 攻击全局结果 ")
    print("="*50)
    print(f"攻击扰动幅度 (Epsilon)  = {EPSILON}")
    print(f"攻击后全局 Accuracy (%)  = {acc_adv * 100:.2f}")
    print(f"攻击后恶意 Precision (%) = {prec_adv * 100:.2f}")
    print(f"攻击后恶意 Recall (%)    = {rec_adv * 100:.2f}")
    print(f"攻击后恶意 F1-Score (%)  = {f1_adv * 100:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()