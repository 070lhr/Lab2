def generate_shap_plots(model, X_test, bg_data, title_prefix, file_suffix):
    model.eval()
    
    print(f"  [*] 正在计算 {title_prefix} 环境的 SHAP 归因，正在执行真实的扰动推断，请稍候...")
    start_time = time.time()
    
    # 核心修改 1：封装黑盒预测函数，彻底摆脱 PyTorch 内部梯度的刚性约束
    def predict_fn(x_numpy):
        with torch.no_grad(): # 绝对禁止梯度干扰
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(DEVICE)
            return model(x_tensor).cpu().numpy()
    
    # 核心修改 2：使用 k-means 聚类提炼背景数据的代表性分布，加速核计算
    # 这一步能让 KernelExplainer 在保持高精度的同时，在几十秒内跑完
    bg_summary = shap.kmeans(bg_data.cpu().numpy(), 15)
    
    # 核心修改 3：使用 KernelExplainer 替代 GradientExplainer
    # 它通过真实的特征掩码扰动来计算边缘贡献，能完美还原数据自身的非对称性和真实方差
    explainer = shap.KernelExplainer(predict_fn, bg_summary)
    
    # 计算 SHAP 值 (silent=True 防止终端打印多余的进度条)
    shap_values = explainer.shap_values(X_test.cpu().numpy(), silent=True)
    
    # KernelExplainer 对单输出模型可能返回 list，这里做兼容解包
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    print(f"  [+] SHAP 计算完成，耗时: {time.time() - start_time:.2f}s")

    # ================= 以下绘图代码保持不变 =================
    try:
        fm.fontManager.addfont(SIMSUN_FONT_PATH)
        fm.fontManager.addfont(TIMES_FONT_PATH)
        simsun_name = fm.FontProperties(fname=SIMSUN_FONT_PATH).get_name()
        times_name = fm.FontProperties(fname=TIMES_FONT_PATH).get_name()
        plt.rcParams['font.sans-serif'] = [times_name, simsun_name]
        plt.rcParams['axes.unicode_minus'] = False 
    except Exception:
        plt.rcParams['axes.unicode_minus'] = False

    # 画柱状图
    plt.figure()
    shap.summary_plot(shap_values, X_test.cpu().numpy(), feature_names=FEATURE_COLS, plot_type="bar", show=False)
    plt.savefig(f"shap_bar_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 画蜂群图
    plt.figure()
    shap.summary_plot(shap_values, X_test.cpu().numpy(), feature_names=FEATURE_COLS, show=False)
    plt.savefig(f"shap_beeswarm_{file_suffix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] 图表已保存为 shap_bar_{file_suffix}.png 和 shap_beeswarm_{file_suffix}.png")
替换后的预期效果