"""
电弧故障检测模型集成模块
整合1D-DITN分类模型和Informer预测模型
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from collections import OrderedDict
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import math

# 添加Informer模型路径
informer_path = Path(__file__).parent / "Arc Prediction Task"
if informer_path.exists():
    sys.path.insert(0, str(informer_path))

try:
    from exp.exp_informer import Exp_Informer
    from utils.tools import dotdict
    INFORMER_AVAILABLE = True
except ImportError:
    INFORMER_AVAILABLE = False
    print("警告: Informer模型模块未找到，预测功能将不可用")

# ==================== 1D-DITN 模型定义 ====================

def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    """PyTorch<1.10 无 padding='same'，手动计算填充"""
    return math.floor(((kernel_size - 1) * dilation) / 2)

class Inception(nn.Module):
    """Inception模块"""
    def __init__(self, input_size, filters, dilation=0):
        super(Inception, self).__init__()
        self.bottleneck = nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding=_same_padding(1),
            bias=False
        )
        
        self.conv1 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=10,
            stride=1,
            padding=_same_padding(10, 1 + dilation),
            dilation=1 + dilation,
            bias=False
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=20,
            stride=1,
            padding=_same_padding(20, 1 + dilation),
            dilation=1 + dilation,
            bias=False
        )
        
        self.conv3 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=40,
            stride=1,
            padding=_same_padding(40, 1 + dilation),
            dilation=1 + dilation,
            bias=False
        )
        
        self.conv4 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding=_same_padding(1),
            bias=False
        )
        
        self.batch_norm = nn.BatchNorm1d(num_features=4 * filters)
    
    def forward(self, x):
        x = self.bottleneck(x)
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y4 = self.conv4(x)
        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.batch_norm(y)
        y = F.relu(y)
        return y


class Residual(nn.Module):
    """残差模块"""
    def __init__(self, input_size, filters):
        super(Residual, self).__init__()
        self.bottleneck = nn.Conv1d(
            in_channels=input_size,
            out_channels=4 * filters,
            kernel_size=1,
            stride=1,
            padding=_same_padding(1),
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4 * filters)
    
    def forward(self, x, y):
        y = y + self.batch_norm(self.bottleneck(x))
        y = F.relu(y)
        return y


class Lambda(nn.Module):
    """Lambda模块"""
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f
    
    def forward(self, x):
        return self.f(x)


class InceptionModel1D(nn.Module):
    """1D-DITN模型（Inception架构）"""
    def __init__(self, input_size, num_classes=2, filters=32, depth=6, dilation=4, dropout=0.5):
        super(InceptionModel1D, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.filters = filters
        self.depth = depth
        self.dilation = dilation
        self.drop = nn.Dropout(p=dropout)
        
        modules = OrderedDict()
        
        for d in range(depth):
            modules[f'inception_{d}'] = Inception(
                input_size=input_size if d == 0 else 4 * filters,
                filters=filters,
                dilation=dilation
            )
            if d % 3 == 2:
                modules[f'residual_{d}'] = Residual(
                    input_size=input_size if d == 2 else 4 * filters,
                    filters=filters
                )
        
        modules['avg_pool'] = Lambda(f=lambda x: torch.mean(x, dim=-1))
        modules['linear1'] = nn.Linear(in_features=4 * filters, out_features=filters)
        modules['linear2'] = nn.Linear(in_features=filters, out_features=num_classes)
        
        self.model = nn.Sequential(modules)

    def forward(self, x):
        # x shape: [batch, channels, seq_len]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, seq_len]
        
        y = None
        for d in range(self.depth):
            y = self.model.get_submodule(f'inception_{d}')(x if d == 0 else y)
            if d % 3 == 2:
                y = self.model.get_submodule(f'residual_{d}')(x, y)
                x = y
        
        y = self.model.get_submodule('avg_pool')(y)
        y = self.model.get_submodule('linear1')(y)
        y = self.drop(F.relu(y))
        y = self.model.get_submodule('linear2')(y)
        return y


# ==================== 模型集成类 ====================

class ArcFaultModelSystem:
    """电弧故障检测模型系统 - 集成1D-DITN和Informer"""
    
    def __init__(self, 
                 ditn_model_path: Optional[str] = None,
                 informer_checkpoint: Optional[str] = None,
                 ditn_config: Optional[Dict] = None,
                 informer_config: Optional[Dict] = None):
        """
        初始化模型系统
        
        Args:
            ditn_model_path: 1D-DITN模型路径
            informer_checkpoint: Informer模型checkpoint路径
            ditn_config: 1D-DITN模型配置
            informer_config: Informer模型配置
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 默认配置
        default_ditn_config = {
            'input_size': 4000,
            'num_classes': 2,  # 二分类：正常/故障
            'filters': 32,
            'depth': 6,
            'dilation': 4,
            'dropout': 0.5
        }
        
        default_informer_config = {
            'model': 'informer',
            'enc_in': 1,
            'dec_in': 1,
            'c_out': 1,
            'seq_len': 48,
            'label_len': 10,
            'pred_len': 20,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 3,
            'd_layers': 2,
            'd_ff': 512,
            'dropout': 0.05,
            'attn': 'prob',
            'factor': 5,
            'output_attention': False,
            'distil': True,
            'mix': True,
            'embed': 'timeF',
            'activation': 'gelu',
            'freq': 's',
            'use_gpu': torch.cuda.is_available(),
            'gpu': 0,
            'use_multi_gpu': False,
            'devices': '0',
            'device_ids': [0],
            'data': 'custom',
            'root_path': str(Path(__file__).parent / "Arc Prediction Task"),
            'data_path': 'predictdate_demo.csv',
            'features': 'S',
            'target': 'value',
            'checkpoints': str(Path(__file__).parent / "Arc Prediction Task" / "checkpoints"),
        }
        
        self.ditn_config = {**default_ditn_config, **(ditn_config or {})}
        class_map_from_config = self.ditn_config.pop('class_map', None)
        self.informer_config = {**default_informer_config, **(informer_config or {})}
        
        # 初始化1D-DITN模型
        self.ditn_model = InceptionModel1D(**self.ditn_config).to(self.device)
        self.ditn_model.eval()
        
        if ditn_model_path and Path(ditn_model_path).exists():
            self.load_ditn_model(ditn_model_path, class_map_override=class_map_from_config)
        else:
            print("警告: 未找到1D-DITN模型文件，使用随机初始化")
            self._update_class_map(class_map_from_config)
        
        # 初始化Informer模型
        self.informer_model = None
        self.informer_exp = None
        if INFORMER_AVAILABLE:
            try:
                informer_args = dotdict(self.informer_config)
                self.informer_exp = Exp_Informer(informer_args)
                # 尝试自动查找checkpoint
                if not informer_checkpoint:
                    # 自动查找默认checkpoint
                    default_checkpoint = Path(__file__).parent / "Arc Prediction Task" / "checkpoints" / "informer_custom_train"
                    if default_checkpoint.exists():
                        informer_checkpoint = str(default_checkpoint)
                
                if informer_checkpoint:
                    if self.load_informer_model(informer_checkpoint):
                        print("✅ Informer模型已加载")
                    else:
                        print("⚠️ Informer模型加载失败，预测功能将不可用")
                else:
                    print("⚠️ 未找到Informer模型checkpoint，预测功能将不可用")
            except Exception as e:
                print(f"Informer模型初始化失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 数据预处理
        self.scaler = MinMaxScaler()
        
        # 类别映射
        self.class_map = self._build_default_class_map(self.ditn_config['num_classes'])
        
        # 统计信息
        self.inference_count = 0
        self.inference_times = []
    
    def load_ditn_model(self, model_path: str, class_map_override: Optional[Dict] = None):
        """加载1D-DITN模型"""
        try:
            state_dict, metadata = self._load_checkpoint_file(model_path)

            checkpoint_config = metadata.get('ditn_config', {})
            if checkpoint_config:
                for key, value in checkpoint_config.items():
                    if key in self.ditn_config:
                        self.ditn_config[key] = value

            checkpoint_num_classes = metadata.get('num_classes')
            if checkpoint_num_classes and checkpoint_num_classes != self.ditn_config['num_classes']:
                self.ditn_config['num_classes'] = checkpoint_num_classes
                self.ditn_model = InceptionModel1D(**self.ditn_config).to(self.device)
                self.ditn_model.eval()

            self.ditn_model.load_state_dict(state_dict)
            print(f"1D-DITN模型已从 {model_path} 加载")

            class_map_data = class_map_override or metadata.get('class_map')
            self._update_class_map(class_map_data)
        except Exception as e:
            print(f"加载1D-DITN模型失败: {e}")
            self._update_class_map(class_map_override)
    
    def load_informer_model(self, checkpoint_path: str):
        """加载Informer模型"""
        if not INFORMER_AVAILABLE or not self.informer_exp:
            return False
        
        try:
            checkpoint_file = Path(checkpoint_path) / 'checkpoint.pth'
            if not checkpoint_file.exists():
                print(f"Informer checkpoint文件不存在: {checkpoint_file}")
                return False
            
            self.informer_exp.model.load_state_dict(
                torch.load(checkpoint_file, map_location=self.device)
            )
            self.informer_model = self.informer_exp.model
            self.informer_model.eval()
            print(f"Informer模型已从 {checkpoint_path} 加载")
            return True
        except Exception as e:
            print(f"加载Informer模型失败: {e}")
            return False
    
    def preprocess_data(self, data: np.ndarray) -> torch.Tensor:
        """数据预处理：确保维度和长度符合模型要求"""
        """简化预处理，适配模拟数据"""
        # 确保数据维度正确 (batch, seq_len)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        # 截断或填充到模型期望的输入长度
        target_length = self.ditn_config['input_size']
        if data.shape[1] > target_length:
            data = data[:, :target_length]
        elif data.shape[1] < target_length:
            data = np.pad(data, ((0, 0), (0, target_length - data.shape[1])), mode='constant')
    
        # 转换为tensor并添加通道维度 (batch, channels, seq_len)
        tensor_data = torch.FloatTensor(data).unsqueeze(1).to(self.device)
        return tensor_data
    
    def classify(self, data: np.ndarray) -> Tuple[str, float, str]:
        """
        使用1D-DITN模型进行分类
        
        Returns:
            (status_text, confidence, fault_type)
        """
        import time
        start_time = time.time()
        
        # 预处理
        tensor_data = self.preprocess_data(data)
        
        # 推理
        with torch.no_grad():
            outputs = self.ditn_model(tensor_data)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        self.inference_count += 1
        
        # 获取结果
        class_idx = predicted.item()
        status_text, fault_type = self.class_map.get(class_idx, ("未知", "unknown"))
        confidence_score = confidence.item() * 100
        
        return status_text, confidence_score, fault_type
    
    def predict(self, data: np.ndarray, seq_len: int = None, pred_len: int = None) -> Optional[np.ndarray]:
        """
        使用Informer模型进行时间序列预测
        
        Args:
            data: 输入时间序列数据
            seq_len: 输入序列长度（默认使用配置值）
            pred_len: 预测序列长度（默认使用配置值）
            
        Returns:
            预测结果，如果模型不可用则返回None
        """
        if not INFORMER_AVAILABLE or self.informer_model is None:
            return None
        
        try:
            seq_len = seq_len or self.informer_config['seq_len']
            pred_len = pred_len or self.informer_config['pred_len']
            label_len = self.informer_config['label_len']
            
            # 准备输入数据
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) == 2 and data.shape[1] > 1:
                # 如果是多列，只取第一列
                data = data[:, 0:1]
            
            # 截取最后seq_len个点
            if data.shape[0] > seq_len:
                input_data = data[-seq_len:].copy()
            else:
                # 填充
                padding = np.zeros((seq_len - data.shape[0], 1))
                input_data = np.concatenate([padding, data], axis=0)
            
            # 转换为tensor
            batch_x = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)  # [1, seq_len, 1]
            
            # 创建时间标记（简化版，使用零填充）
            # 实际应用中应该使用真实的时间特征
            batch_x_mark = torch.zeros(1, seq_len, 4).to(self.device)  # [1, seq_len, 4]
            batch_y_mark = torch.zeros(1, pred_len, 4).to(self.device)  # [1, pred_len, 4]
            
            # decoder输入：label_len的历史数据 + pred_len的零填充
            dec_inp = torch.zeros(1, label_len + pred_len, 1).to(self.device)
            # 使用最后label_len个点作为decoder的起始输入
            if input_data.shape[0] >= label_len:
                dec_inp[:, :label_len, :] = torch.FloatTensor(input_data[-label_len:]).unsqueeze(0).to(self.device)
            else:
                dec_inp[:, :input_data.shape[0], :] = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                pred = self.informer_model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
            
            # 只返回预测部分（最后pred_len个点）
            return pred.cpu().numpy().squeeze()
        except Exception as e:
            print(f"Informer预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def inference(self, data: np.ndarray, fault_scenario: str = None) -> Tuple[str, float, str]:
        """
        综合推理：分类 + 预测
        
        Args:
            data: 输入数据
            fault_scenario: 故障场景（用于模拟模式）
            
        Returns:
            (status_text, confidence, fault_type)
        """
        # 使用1D-DITN进行分类
        status_text, confidence, fault_type = self.classify(data)
        
        # 如果检测到故障，使用Informer进行预测
        if fault_type == "fault" and self.informer_model is not None:
            prediction = self.predict(data)
            if prediction is not None:
                # 可以根据预测结果调整置信度
                # 这里简化处理
                pass
        
        # 如果是模拟模式，可以根据fault_scenario调整结果
        if fault_scenario:
            if fault_scenario == "severe_arc":
                status_text, confidence, fault_type = "二级预警 (故障确认)", 97.5, "severe_arc"
            elif fault_scenario == "early_arc":
                if confidence < 70:
                    status_text, confidence, fault_type = "一级预警 (预测风险)", min(90.0, confidence + 20), "early_arc"
            elif fault_scenario == "motor_start":
                status_text, confidence, fault_type = "干扰信号 (电机启动)", 10.0, "motor_start"
            elif fault_scenario == "normal":
                status_text, confidence, fault_type = "运行正常 (安全)", max(2.0, confidence), "normal"
        
        return status_text, confidence, fault_type
    
    def get_statistics(self) -> Dict:
        """获取模型统计信息"""
        avg_inference_time = np.mean(self.inference_times[-100:]) if self.inference_times else 0
        
        stats = {
            "ditn_model": "已加载" if self.ditn_model else "未加载",
            "informer_model": "已加载" if self.informer_model else "未加载",
            "total_inferences": self.inference_count,
            "average_inference_time_ms": round(avg_inference_time, 2),
            "device": str(self.device),
            "num_classes": self.ditn_config.get('num_classes', 0)
        }
        
        return stats

    def _load_checkpoint_file(self, model_path: str):
        """加载checkpoint，返回(state_dict, metadata)"""
        checkpoint = torch.load(model_path, map_location=self.device)
        metadata: Dict = {}
        state_dict = checkpoint

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']

            for key in ('num_classes', 'class_map', 'ditn_config'):
                if key in checkpoint:
                    metadata[key] = checkpoint[key]

        return state_dict, metadata

    def _build_default_class_map(self, num_classes: int) -> Dict[int, Tuple[str, str]]:
        """根据类别数量生成默认映射"""
        class_map: Dict[int, Tuple[str, str]] = {}
        for idx in range(num_classes):
            if idx == 0:
                class_map[idx] = ("运行正常 (安全)", "normal")
            elif idx == 1 and num_classes == 2:
                class_map[idx] = ("故障预警 (异常)", "fault")
            else:
                class_map[idx] = (f"类别 {idx}", f"class_{idx}")
        return class_map

    def _update_class_map(self, class_map_data: Optional[Dict]):
        """更新类别映射"""
        if not class_map_data:
            self.class_map = self._build_default_class_map(self.ditn_config['num_classes'])
            return

        formatted: Dict[int, Tuple[str, str]] = {}
        for key, value in class_map_data.items():
            try:
                idx = int(key)
            except Exception:
                continue

            status_text = None
            fault_type = None

            if isinstance(value, (list, tuple)):
                if len(value) >= 2:
                    status_text, fault_type = value[0], value[1]
                elif len(value) == 1:
                    status_text = value[0]
            elif isinstance(value, dict):
                status_text = value.get("status") or value.get("text") or value.get("label")
                fault_type = value.get("fault_type") or value.get("code")
            elif isinstance(value, str):
                status_text = value

            status_text = status_text or f"类别 {idx}"
            fault_type = fault_type or f"class_{idx}"
            formatted[idx] = (status_text, fault_type)

        default_map = self._build_default_class_map(self.ditn_config['num_classes'])
        for idx in range(self.ditn_config['num_classes']):
            if idx not in formatted:
                formatted[idx] = default_map.get(idx, (f"类别 {idx}", f"class_{idx}"))

        self.class_map = formatted

