import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ================= 配置 =================
# 1. 模型与数据路径
MODEL_PATH = './dpg_net_model.pth'
SCALER_PATH = './dpg_scaler.pkl'
FLASH_FILE = './flash_event_9dim_full.csv' # 用您过滤后的高质量 Flash
DDOS_FILE = './ciciot_ddos_9dim_full.csv'

# 2. 输出：生成的对抗样本保存路径
ADV_OUTPUT_CSV = './adversarial_ddos_samples.csv'

# 3. 攻击强度 (Epsilon)
# 这个值越大，扰动越强，DDoS 越像 Flash，模型越容易瞎
EPSILON = 0.2  

# 4. 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================

# --- 模型定义 (必须与训练时一致) ---
class DPG_Net(nn.Module):
    def __init__(self):
        super(DPG_Net, self).__init__()
        # 动力学流
        self.dynamics_branch = nn.Sequential(
            nn.Linear(3, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        # 分布流
        self.dist_branch = nn.Sequential(
            nn.Linear(6, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(16 + 32, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, training_mode=False):
        # 攻击时不需要 dropout，我们需要拿到准确的梯度
        x_dyn = x[:, 0:3]
        x_dist = x[:, 3:9]
        out_dyn = self.dynamics_branch(x_dyn)
        out_dist = self.dist_branch(x_dist)
        combined = torch.cat((out_dyn, out_dist), dim=1)
        logits = self.fusion_layer(combined)
        return self.sigmoid(logits)

# --- FGSM 核心函数 ---
def fgsm_attack(model, data, target, epsilon):
    """
    生成对抗样本
    data: 原始特征 (Tensor)
    target: 真实标签 (Tensor)
    epsilon: 扰动大小
    """
    # 1. 允许计算输入数据的梯度
    data.requires_grad = True
    
    # 2. 前向传播
    output = model(data)
    
    # 3. 计算 Loss (我们希望 Loss 变大，即让模型预测错)
    # 这是一个 trick: 我们希望模型把 DDoS(1) 预测成 Flash(0)
    # 所以我们计算它与 "Flash(0)" 的距离，然后去最小化这个距离？
    # 不，标准 FGSM 是 maximize Loss(model(x), true_label)
    loss = nn.BCELoss()(output, target)
    
    # 4. 梯度归零并反向传播
    model.zero_grad()
    loss.backward()
    
    # 5. 获取输入的梯度符号
    data_grad = data.grad.data.sign()
    
    # 6. 生成对抗样本 (沿着梯度上升的方向走，让 Loss 变大)
    # x_adv = x + epsilon * sign(grad)
    perturbed_data = data + epsilon * data_grad
    
    return perturbed_data

def generate_adversarial():
    print("[*] 正在加载环境...")
    
    # 1. 加载 Scaler
    if not os.path.exists(SCALER_PATH):
        print("[!] 找不到标准化器 (.pkl)，请先运行训练脚本。")
        return
    scaler = joblib.load(SCALER_PATH)
    
    # 2. 加载数据 (只加载 DDoS 数据用来攻击)
    df_ddos = pd.read_csv(DDOS_FILE)
    # 排除非特征列
    feature_cols = [
        'Rate', 'Rate_Accel', 'Rate_CV', 
        'Entropy', 'Ent_Change', 'Ent_MA', 
        'Size_Std', 'SizeStd_Change', 'SizeStd_MA'
    ]
    X_ddos = df_ddos[feature_cols].values
    y_ddos = np.ones(len(df_ddos)) # Label = 1
    
    # 标准化 (非常重要！攻击必须在标准化空间进行)
    X_ddos_norm = scaler.transform(X_ddos)
    
    # 转 Tensor
    X_tensor = torch.tensor(X_ddos_norm, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y_ddos, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    # 3. 加载模型
    model = DPG_Net().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # 评估模式
    
    # 4. 测试攻击前的准确率
    print("\n[*] 攻击前基准测试 (Clean Accuracy)...")
    with torch.no_grad():
        clean_out = model(X_tensor)
        clean_pred = (clean_out > 0.5).float()
        clean_acc = (clean_pred.eq(y_tensor).sum() / y_tensor.shape[0]).item()
        print(f"    DDoS 识别率: {clean_acc*100:.2f}% (应该是 100%)")

    # 5. 执行 FGSM 攻击
    print(f"\n[*] 开始生成对抗样本 (FGSM Attack, Epsilon={EPSILON})...")
    # 这里的逻辑是：让模型无法识别这些是 DDoS
    X_adv_tensor = fgsm_attack(model, X_tensor, y_tensor, EPSILON)
    
    # 6. 测试攻击后的准确率
    print("[*] 攻击后鲁棒性测试 (Adversarial Accuracy)...")
    with torch.no_grad():
        adv_out = model(X_adv_tensor)
        adv_pred = (adv_out > 0.5).float() # 依然预测它是 1
        adv_acc = (adv_pred.eq(y_tensor).sum() / y_tensor.shape[0]).item()
        print(f"    DDoS 识别率: {adv_acc*100:.2f}% (越低说明攻击越成功)")
        print(f"    --> 欺骗成功率: {(1-adv_acc)*100:.2f}%")

    # 7. 逆标准化并保存对抗样本
    # 我们需要把扰动后的数据还原成真实数值，看看它变成了什么样
    X_adv_np = X_adv_tensor.cpu().detach().numpy()
    X_adv_original_scale = scaler.inverse_transform(X_adv_np)
    
    # 创建 DataFrame
    df_adv = pd.DataFrame(X_adv_original_scale, columns=feature_cols)
    df_adv['Label'] = 1 # 它们本质上还是 DDoS
    
    # 保存
    df_adv.to_csv(ADV_OUTPUT_CSV, index=False)
    print(f"\n[*] 对抗样本已保存至: {ADV_OUTPUT_CSV}")
    print("    您可以打开查看，主要观察 'Size_Std' 和 'Entropy' 是否变得像 Flash 了。")

if __name__ == "__main__":
    generate_adversarial()