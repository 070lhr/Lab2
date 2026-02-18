import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ================= 配置 =================
MODEL_PATH = './dpg_net_model.pth'
SCALER_PATH = './dpg_scaler.pkl'
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
ADV_OUTPUT_CSV = './adversarial_ddos_samples.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================

# --- 模型定义 ---
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
            # 攻击时不需要 Dropout
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_dyn = x[:, 0:3]
        x_dist = x[:, 3:9]
        out_dyn = self.dynamics_branch(x_dyn)
        out_dist = self.dist_branch(x_dist)
        combined = torch.cat((out_dyn, out_dist), dim=1)
        logits = self.fusion_layer(combined)
        return self.sigmoid(logits)

# --- 增强版 FGSM ---
def fgsm_attack_batch(model, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    
    # 使用 BCELoss
    loss = nn.BCELoss()(output, target)
    model.zero_grad()
    loss.backward()
    
    # 【诊断】检查梯度是否为 0
    grad_mag = data.grad.data.abs().mean().item()
    if grad_mag < 1e-6:
        print(f"    [警告] 梯度消失 (Mag={grad_mag:.6f})! 模型太自信了，FGSM 推不动。")
        # 这种情况下，对抗样本就是原始样本
        return data.detach()
    
    data_grad = data.grad.data.sign()
    
    # 生成对抗样本
    # DDoS(1) -> 伪装成 Flash(0)，我们需要 minimize output probability
    # 但 FGSM 是沿着梯度上升让 Loss 变大。
    # Loss 变大 = 预测结果远离 Target(1) = 预测结果接近 0
    # 所以公式没问题：x + eps * sign(grad)
    perturbed_data = data + epsilon * data_grad
    return perturbed_data.detach()

def main():
    print("[*] 正在加载环境...")
    scaler = joblib.load(SCALER_PATH)
    
    # 读取数据
    df_ddos = pd.read_csv(DDOS_FILE)
    # 稍微采样一点，不用全部跑，跑前 5000 个够了
    df_ddos = df_ddos.head(5000)
    
    feature_cols = [
        'Rate', 'Rate_Accel', 'Rate_CV', 
        'Entropy', 'Ent_Change', 'Ent_MA', 
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA'
    ]
    X_ddos = df_ddos[feature_cols].values
    y_ddos = np.ones(len(df_ddos)) 
    
    X_ddos_norm = scaler.transform(X_ddos)
    X_tensor = torch.tensor(X_ddos_norm, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y_ddos, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    model = DPG_Net().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 基准测试
    with torch.no_grad():
        clean_out = model(X_tensor)
        clean_acc = ((clean_out > 0.5).float().eq(y_tensor).sum() / y_tensor.shape[0]).item()
        print(f"[*] 基准准确率: {clean_acc*100:.2f}%")

    # === 暴力搜索最佳 Epsilon ===
    # 既然 0.2 不行，我们就试大一点
    epsilons = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    
    best_adv_data = None
    lowest_acc = 1.0
    
    print(f"\n[*] 开始暴力搜索最佳攻击强度...")
    print(f"{'Epsilon':<10} | {'Model Acc':<15} | {'Success Rate':<15}")
    print("-" * 45)
    
    for eps in epsilons:
        # 重新加载 graph
        X_temp = X_tensor.clone().detach()
        y_temp = y_tensor.clone().detach()
        
        # 攻击
        X_adv = fgsm_attack_batch(model, X_temp, y_temp, eps)
        
        # 测试效果
        with torch.no_grad():
            out = model(X_adv)
            acc = ((out > 0.5).float().eq(y_temp).sum() / y_temp.shape[0]).item()
            success_rate = 1.0 - acc
            
            print(f"{eps:<10} | {acc*100:.2f}%          | {success_rate*100:.2f}%")
            
            # 保存效果最好（准确率最低）的那一组
            if acc < lowest_acc:
                lowest_acc = acc
                best_adv_data = X_adv
                
            # 如果成功率已经很高了（比如 > 80%），就没必要再加大了
            if success_rate > 0.8:
                print("    -> 攻击成功率达标，停止搜索。")
                break

    # === 保存最佳对抗样本 ===
    if best_adv_data is not None:
        X_adv_np = best_adv_data.cpu().numpy()
        X_adv_original = scaler.inverse_transform(X_adv_np)
        
        df_adv = pd.DataFrame(X_adv_original, columns=feature_cols)
        df_adv['Label'] = 1 # 依然标记为 DDoS，用于后续对抗训练
        df_adv.to_csv(ADV_OUTPUT_CSV, index=False)
        print(f"\n[*] 已保存最强对抗样本 (Acc={lowest_acc*100:.2f}%) 至: {ADV_OUTPUT_CSV}")
        print("    -> 现在您可以把这些样本加入训练集，进行对抗训练了！")
    else:
        print("\n[!] 警告：所有攻击都失败了。您的模型可能是无敌的，或者梯度完全消失。")

if __name__ == "__main__":
    main()