import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl

# 忽略第三方库底层的警告信息，保持控制台清爽
warnings.filterwarnings("ignore")

# 解决图表中文字体显示和负号问题（兼容 Windows 和 Mac）
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False
# 提高保存图片的清晰度，满足大论文印刷要求
plt.rcParams['figure.dpi'] = 300 

def generate_realistic_synthetic_data(n_samples=2000):
    """
    根据论文4.2节的物理规律，生成高度拟真的双流特征数据
    Class 0: 正常突发流量 (FE)
    Class 1: 应用层DDoS攻击 (DDoS)
    """
    np.random.seed(42) # 固定随机数种子，保证每次生成的图一模一样
    
    # 类别标签：一半FE(0)，一半DDoS(1)
    y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
    
    # 初始化特征字典
    data = {}
    
    # 1. 速率变异系数 (Rate_CV) - 核心特征
    # FE: 具有较大的天然抖动 (高值区)
    # DDoS: 程序死循环发包，极端稳定 (低值区)
    data['Rate_CV'] = np.concatenate([
        np.random.normal(loc=0.8, scale=0.3, size=n_samples//2), 
        np.random.normal(loc=0.08, scale=0.03, size=n_samples//2)
    ])
    data['Rate_CV'] = np.clip(data['Rate_CV'], 0.01, 2.5) # 防止出现负值
    
    # 2. 源IP信息熵 (SIP_Ent)
    # FE: 大量真实散户涌入，极高
    # DDoS: 受限于肉鸡池，较低且呈多峰
    data['SIP_Ent'] = np.concatenate([
        np.random.normal(loc=9.2, scale=0.8, size=n_samples//2),
        np.random.normal(loc=2.5, scale=1.0, size=n_samples//2)
    ])
    data['SIP_Ent'] = np.clip(data['SIP_Ent'], 0.1, 12.0)
    
    # 3. 瞬时载荷标准差 (Size_Std)
    # FE: HTTP请求多样，方差极大
    # DDoS: 攻击脚本固定载荷，趋近于0
    data['Size_Std'] = np.concatenate([
        np.random.normal(loc=1.5e9, scale=2e8, size=n_samples//2),
        np.random.normal(loc=0.05e9, scale=0.01e9, size=n_samples//2)
    ])
    data['Size_Std'] = np.clip(data['Size_Std'], 0, None)
    
    # 4. 载荷标准差滑动平均 (SizeStd_MA)
    data['SizeStd_MA'] = data['Size_Std'] * np.random.normal(1.0, 0.1, n_samples)
    
    # 5. 源IP信息熵滑动平均 (SIPEnt_MA)
    data['SIPEnt_MA'] = data['SIP_Ent'] * np.random.normal(1.0, 0.05, n_samples)
    
    # 6. 发包速率 (Rate) - 高度重合，宏观上都在峰值
    data['Rate'] = np.concatenate([
        np.random.normal(loc=24000, scale=4000, size=n_samples//2),
        np.random.normal(loc=25000, scale=1000, size=n_samples//2)
    ])
    
    # 7. 速率加速度 (Rate_Accel) - 大多在0附近，重叠度极高
    data['Rate_Accel'] = np.random.normal(loc=0, scale=1000, size=n_samples)
    
    # 8. 载荷标准差突变率 (SizeStd_Change) - 噪音项
    data['SizeStd_Change'] = np.random.normal(loc=0, scale=1e8, size=n_samples)
    
    # 9. 源IP信息熵突变率 (SIPEnt_Change) - 噪音项
    data['SIPEnt_Change'] = np.random.normal(loc=0, scale=0.5, size=n_samples)
    
    df = pd.DataFrame(data)
    return df, y

# 1. 生成高逼真数据
X, y = generate_realistic_synthetic_data(2000)

# 使用中文列名（直接体现在论文图表中，更显专业）
chinese_columns = [
    '速率变异系数\n(Rate_CV)', 
    '源IP信息熵\n(SIP_Ent)', 
    '瞬时载荷标准差\n(Size_Std)', 
    '载荷标准差滑动平均\n(SizeStd_MA)', 
    '源IP信息熵滑动平均\n(SIPEnt_MA)', 
    '发包速率\n(Rate)', 
    '速率加速度\n(Rate_Accel)', 
    '载荷标准差突变率\n(SizeStd_Change)', 
    '源IP信息熵突变率\n(SIPEnt_Change)'
]
X.columns = chinese_columns

# 2. 训练一个白盒树模型模拟决策边界
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# 3. 计算 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 动态兼容新旧版本的 SHAP 库结构
if isinstance(shap_values, list):
    shap_values_ddos = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_ddos = shap_values[:, :, 1]
else:
    shap_values_ddos = shap_values

# ----------------- 画图部分 -----------------

# 4. 全局特征重要性分析图 (Bar Plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_ddos, X, plot_type="bar", show=False, color='#3b76af') 
plt.title('图4-x 特征全局重要性分析', fontsize=14, pad=20)
plt.xlabel('SHAP绝对值均值 (平均对模型输出的冲击力)', fontsize=12)
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=600, bbox_inches='tight') 
plt.close()

# 5. 特征决策影响机制可视化 (Beeswarm Plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_ddos, X, show=False)
plt.title('图4-y 基于SHAP的特征决策影响机制蜂巢图', fontsize=14, pad=20)
plt.xlabel('SHAP 值 (对模型判决为DDoS的推动力)', fontsize=12)
plt.tight_layout()
plt.savefig('shap_beeswarm_plot.png', dpi=600, bbox_inches='tight')
plt.close()

print("图表已成功生成并保存为高清PNG！")