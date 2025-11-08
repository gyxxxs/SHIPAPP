"""
模型配置示例文件

说明如何配置1D-DITN和Informer模型的路径
"""

# ==================== 模型路径配置 ====================

# 1D-DITN模型路径（二分类模型）
# 如果模型已训练并保存，指定路径
DITN_MODEL_PATH = None  # 例如: "Arc Classification Task/Two_class/checkpoints/model.pth"

# Informer模型checkpoint路径
# 训练后的模型会保存在 checkpoints/informer_custom_train/checkpoint.pth
INFORMER_CHECKPOINT = None  # 例如: "Arc Prediction Task/checkpoints/informer_custom_train"

# ==================== 使用示例 ====================

# 在 zhinengti.py 中修改 get_model_diagnostics() 函数：

"""
@st.cache_resource
def get_model_diagnostics():
    '''获取模型诊断实例'''
    if MODEL_DIAGNOSTICS_AVAILABLE:
        return ModelDiagnostics(
            ditn_model_path=DITN_MODEL_PATH,
            informer_checkpoint=INFORMER_CHECKPOINT
        )
    return None
"""

# ==================== 模型训练说明 ====================

# 1. 训练1D-DITN模型：
#    - 打开 Arc Classification Task/Two_class/Two_class_Train_Files/1D-DITN/two-1D-DITN.ipynb
#    - 运行训练代码
#    - 保存模型为 .pth 文件
#    - 将路径设置为 DITN_MODEL_PATH

# 2. 训练Informer模型：
#    - 进入 Arc Prediction Task 目录
#    - 运行: python train.py
#    - 模型会自动保存到 checkpoints/informer_custom_train/checkpoint.pth
#    - 将路径设置为 INFORMER_CHECKPOINT

# ==================== 自动检测模式 ====================

# 如果不指定路径，系统会：
# 1. 尝试自动查找模型文件
# 2. 如果找不到，使用模拟模型（不影响功能）
# 3. 在侧边栏显示模型状态

