import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import sys

# ================= 配置 =================
MODEL_PATH = './dpg_net_model.pth'
SCALER_PATH = './dpg_scaler.pkl'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
ADV_OUTPUT_CSV = './adversarial_ddos_samples.csv'
EPSILONS_TO_SEARCH = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================

# --- 1. 模型定义 (必须完全一致) ---
class DPG_Net(nn.Module):
    def __init__(self):
        super(DPG_Net, self).__init__()
        self.dynamics_branch = nn.Sequential(
            nn.Linear(3, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.dist_branch = nn.Sequential(
            nn.Linear(6, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(16 + 32, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    # === 修改点：增加一个只返回 logits 的方法 ===
    # 这样我们可以绕过 Sigmoid 直接攻击
    def forward_logits(self, x):
        x_dyn = x[:, 0:3]
        x_dist = x[:, 3:9]
        out_dyn = self.dynamics_branch(x_dyn)
        out_dist = self.dist_branch(x_dist)
        combined = torch.cat((out_dyn, out_dist), dim=1)
        logits = self.fusion_layer(combined)
        return logits # 注意：这里没有 Sigmoid

    def forward(self, x):
        return self.sigmoid(self.forward_logits(x))

# --- 2. 基于 Logits 的 FGSM 攻击 ---
def fgsm_attack_logits(model, data, target, epsilon):
    """
    使用 BCEWithLogitsLoss 进行攻击，避免梯度消失
    """
    data_copy = data.clone().detach().requires_grad_(True)
    
    # 1. 获取 Logits (不经过 Sigmoid)
    logits = model.forward_logits(data_copy)
    
    # 2. 计算 Loss (BCEWithLogitsLoss 更稳定)
    # 我们希望 Loss 变大 (让模型预测错)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, target)
    
    # 3. 反向传播
    model.zero_grad()
    loss.backward()
    
    # 4. 获取梯度
    data_grad = data_copy.grad.data
    
    # 【诊断】打印梯度的平均强度，看看是不是 0
    # grad_mean = data_grad.abs().mean().item()
    # if grad_mean == 0:
    #     print(" [!] 梯度依然是 0，这很不正常！")
    
    # 5. 生成对抗样本
    # x_adv = x + epsilon * sign(grad)
    perturbed_data = data + epsilon * data_grad.sign()
    
    return perturbed_data.detach()

# --- 3. 主程序 ---
def main():
    print(f"[*] 初始化 (Device: {DEVICE})...")
    
    if not os.path.exists(MODEL_PATH): return
    scaler = joblib.load(SCALER_PATH)
    
    # 读取数据 (取前 5000 条)
    df_ddos = pd.read_csv(DDOS_FILE).head(5000)
    
    feature_cols = [
        'Rate', 'Rate_Accel', 'Rate_CV', 
        'Entropy', 'Ent_Change', 'Ent_MA', 
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA'
    ]
    
    X = df_ddos[feature_cols].values
    y = np.ones(len(df_ddos)) 
    
    X_norm = scaler.transform(X)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    model = DPG_Net().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # === 暴力搜索 ===
    print(f"\n[*] --- 使用 Logits 进行强力攻击 ---")
    print(f"{'Epsilon':<10} | {'Model Acc':<15} | {'Diff (L2)':<15}")
    print("-" * 50)
    
    lowest_acc = 1.0
    best_adv = None
    best_eps = 0
    
    for eps in EPSILONS_TO_SEARCH:
        # 攻击
        X_adv = fgsm_attack_logits(model, X_tensor, y_tensor, eps)
        
        # 检查数据到底变了没？(计算 L2 距离)
        diff = (X_adv - X_tensor).norm().item()
        
        # 测试效果 (用完整的 forward 测准确率)
        with torch.no_grad():
            probs = model(X_adv)
            acc = ((probs > 0.5).float().eq(y_tensor).sum() / y_tensor.shape[0]).item()
            
            print(f"{eps:<10} | {acc*100:.2f}%          | {diff:.4f}")
            
            if acc < lowest_acc:
                lowest_acc = acc
                best_adv = X_adv
                best_eps = eps

    # 保存
    if best_adv is not None:
        print("\n" + "="*50)
        print(f"[*] 最强攻击强度: Epsilon={best_eps}")
        print(f"[*] 此时模型识别率: {lowest_acc*100:.2f}%")
        
        X_adv_np = best_adv.cpu().numpy()
        X_ori = scaler.inverse_transform(X_adv_np)
        df_adv = pd.DataFrame(X_ori, columns=feature_cols)
        df_adv['Label'] = 1 
        
        df_adv.to_csv(ADV_OUTPUT_CSV, index=False)
        print(f"[*] 对抗样本已保存至: {ADV_OUTPUT_CSV}")
    else:
        print("[!] 居然还是推不动？可能是 Epsilon 依然太小，或者模型出现了数值溢出。")

if __name__ == "__main__":
    main()