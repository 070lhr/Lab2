import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# ================= 配置区域 =================
DPG_MODEL_PATH = './dpg_net_model.pth' # 您第四章的双动力模型
SCALER_PATH = './dpg_scaler.pkl'
FLASH_FILE = './flash_event_9dim_full.csv' 
DDOS_FILE = './ciciot_ddos_9dim_full.csv'
OUTPUT_ROC_IMG = 'roc_comparison_dpgnet.png'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

# --- 第四章模型定义：DPG-Net (双流动力学感知网络) ---
class DPG_Net(nn.Module):
    def __init__(self):
        super(DPG_Net, self).__init__()
        # 支路 1: 动力学特征 (双动力之一)
        self.dynamics_branch = nn.Sequential(
            nn.Linear(3, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        # 支路 2: 分布特征 (双动力之二)
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

    def forward(self, x):
        x_dyn = x[:, 0:3]
        x_dist = x[:, 3:9]
        out_dyn = self.dynamics_branch(x_dyn)
        out_dist = self.dist_branch(x_dist)
        combined = torch.cat((out_dyn, out_dist), dim=1)
        logits = self.fusion_layer(combined)
        return self.sigmoid(logits)

def main():
    print("[*] 正在加载数据，准备第四章对比实验...")
    scaler = joblib.load(SCALER_PATH)
    
    # 1. 加载并合并数据
    df_flash = pd.read_csv(FLASH_FILE)
    df_ddos = pd.read_csv(DDOS_FILE)
    
    feature_cols = [c for c in df_flash.columns if c not in ['Label', 'timestamp', 'Unnamed: 0']]
    
    X_raw = np.concatenate([df_flash[feature_cols].values, df_ddos[feature_cols].values], axis=0)
    y = np.concatenate([np.zeros(len(df_flash)), np.ones(len(df_ddos))], axis=0)
    
    X = scaler.transform(X_raw)
    
    # 划分 20% 作为纯净测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. 训练传统机器学习模型 (作为基线靶子)
    print("[*] 正在训练 Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    print("[*] 正在训练 SVM (由于复杂度高，采用 2 万条采样训练)...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42) 
    svm_model.fit(X_train[:20000], y_train[:20000]) 
    svm_probs = svm_model.predict_proba(X_test)[:, 1]

    # 3. 加载您第四章的 DPG-Net 模型
    print("[*] 正在测试 DPG-Net (Ours)...")
    dpg_net = DPG_Net().to(DEVICE)
    dpg_net.load_state_dict(torch.load(DPG_MODEL_PATH, map_location=DEVICE))
    dpg_net.eval()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        dpg_probs = dpg_net(X_test_tensor).cpu().numpy().flatten()

    # 4. 计算 ROC 和 AUC
    print("\n[*] 正在计算并绘制 ROC 曲线...")
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    auc_rf = auc(fpr_rf, tpr_rf)
    
    fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
    auc_svm = auc(fpr_svm, tpr_svm)
    
    fpr_dpg, tpr_dpg, _ = roc_curve(y_test, dpg_probs)
    auc_dpg = auc(fpr_dpg, tpr_dpg)

    # 5. 绘图 (顶级会议风格)
    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    plt.plot(fpr_rf, tpr_rf, color='#2ca02c', lw=2.5, linestyle='-.', 
             label=f'Random Forest (AUC = {auc_rf:.4f})')
    plt.plot(fpr_svm, tpr_svm, color='#ff7f0e', lw=2.5, linestyle='--', 
             label=f'SVM (AUC = {auc_svm:.4f})')
    # 突出显示您的模型
    plt.plot(fpr_dpg, tpr_dpg, color='#d62728', lw=3, linestyle='-', 
             label=f'DPG-Net (Ours) (AUC = {auc_dpg:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (误报率)', fontsize=13)
    plt.ylabel('True Positive Rate (召回率)', fontsize=13)
    plt.title('ROC Curve Comparison (DDoS vs Flash Crowd)', fontsize=15, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_ROC_IMG)
    print(f"\n[OK] 绘图完成！图片已保存至: {OUTPUT_ROC_IMG}")

if __name__ == "__main__":
    main()