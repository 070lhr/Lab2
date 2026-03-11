#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from matplotlib.font_manager import FontProperties

# ================= 配置区域 =================
# 1. 输入数据路径 (您上一步提取出的混合 3 维特征文件)
INPUT_CSV = './tinubu_3dim_full_mixed.csv'

# 2. 输出图表路径
OUTPUT_CM_IMG = 'tinubu_dt_confusion_matrix.png'
OUTPUT_ROC_IMG = 'tinubu_dt_roc_curve.png'

# 3. 指定字体 (防止画图时中文字符显示为方块，请确保路径正确或使用系统默认英文)
# my_font = FontProperties(fname='./MSYH.TTC', size=12) 
# 如果 Linux 上没有中文字体，我们直接用英文作图，更符合顶级论文规范
# ===========================================

def main():
    print("[*] 正在加载 Tinubu 3 维特征混合数据集...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"[!] 找不到文件 {INPUT_CSV}，请检查路径！")
        return

    # 1. 数据准备
    # 特征列：源IP数，新源IP数，平均到达时间间隔
    X = df[['Num_SrcIP', 'Num_New_SrcIP', 'Mean_IAT']]
    y = df['Label']  # 1: DDoS, 0: FE

    # 按照 8:2 划分训练集和测试集，并使用 stratify 保证类别比例均衡
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[*] 数据集划分完成:")
    print(f"    - 训练集样本数: {len(X_train)} (DDoS: {sum(y_train==1)}, FE: {sum(y_train==0)})")
    print(f"    - 测试集样本数: {len(X_test)} (DDoS: {sum(y_test==1)}, FE: {sum(y_test==0)})\n")

    # 2. 原汁原味复现 Tinubu 的 DT-Model (C4.5 算法)
    # 核心细节：criterion='entropy' 完全对应原论文基于信息熵/信息增益的 C4.5 逻辑
    print("[*] 正在训练 DT-Model (Criterion = Entropy)...")
    dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt_model.fit(X_train, y_train)
    print("[+] 模型训练完毕！\n")

    # 3. 在洁净测试集上进行推理评估 (对应 4.5.2 节)
    y_pred = dt_model.predict(X_test)
    y_prob = dt_model.predict_proba(X_test)[:, 1] # 获取预测为 1 (DDoS) 的概率，用于画 ROC

    # 4. 计算并打印可以直接填入论文表格的学术指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    error_rate = 1.0 - acc

    print("="*50)
    print(" 🌟 洁净流量场景下的检测性能评估结果 (DT 3D) 🌟")
    print("="*50)
    print(f" Accuracy (准确率)  : {acc * 100:.2f}%")
    print(f" Precision(精确率)  : {prec * 100:.2f}%")
    print(f" Recall   (召回率)  : {rec * 100:.2f}%")
    print(f" F1-Score (F1分数)  : {f1 * 100:.2f}%")
    print(f" Error Rate(误报率) : {error_rate * 100:.2f}%")
    print("="*50)
    print("\n[详细分类报告]:")
    print(classification_report(y_test, y_pred, target_names=['Flash Event (0)', 'DDoS (1)']))

    # 5. 绘制混淆矩阵 (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['FE', 'DDoS'], yticklabels=['FE', 'DDoS'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('DT-Model Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_CM_IMG, dpi=300)
    print(f"[*] 混淆矩阵已保存至: {OUTPUT_CM_IMG}")

    # 6. 绘制 ROC 曲线并计算 AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (DT-Model)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROC_IMG, dpi=300)
    print(f"[*] ROC 曲线已保存至: {OUTPUT_ROC_IMG}")
    print("="*50)

if __name__ == "__main__":
    main()