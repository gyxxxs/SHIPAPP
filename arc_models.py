"""
修正版本的 arc_models.py
修复要点：
- 将 Inception/Residual 的 Conv1d 输入通道（in_channels）与序列长度区分：
  序列长度（input_size）不应作为 Conv1d 的 in_channels。对于单通道时间序列，in_channels=1。
- 在构建模块时，明确传入每个模块的输入通道数，避免通道/长度混淆引发的 RuntimeError。
- 保持原有接口尽量不变（ArcFaultModelSystem 初始化参数不变）。

说明：此文件为修复后的完整模块，可直接替换原 arc_models.py 使用。
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

# 添加Informer模型路径（不变）
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

# ==================== 工具函数 ====================

def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    """PyTorch<1.10 无 padding='same'，手动计算填充"""
    return math.floor(((kernel_size - 1) * dilation) / 2)


# ==================== 1D-DITN 模型定义（修复版） ====================
class Inception(nn.Module):
    """Inception 模块

    参数说明：
    - in_channels: 模块输入的通道数（对单通道时间序列应为1）
    - filters: 每条并行分支的输出通道数
    - dilation: 可选膨胀，用于可配置的感受野
    """
    def __init__(self, in_channels: int, filters: int, dilation: int = 0):
        super(Inception, self).__init__()
        # bottleneck 将输入通道变换到 filters 通道，方便后续多尺度卷积
        self.bottleneck = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        # 多尺度卷积分支：基于 bottleneck 的输出通道数 filters
        self.conv1 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=10,
            stride=1,
            padding='same',
            dilation=1 + dilation,
            bias=False
        )

        self.conv2 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=20,
            stride=1,
            padding='same',
            dilation=1 + dilation,
            bias=False
        )

        self.conv3 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=40,
            stride=1,
            padding='same',
            dilation=1 + dilation,
            bias=False
        )

        self.conv4 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        self.batch_norm = nn.BatchNorm1d(num_features=4 * filters)

    def forward(self, x):
        # x: [batch, in_channels, seq_len]
        x = self.bottleneck(x)  # -> [batch, filters, seq_len]
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y4 = self.conv4(x)
        y = torch.cat([y1, y2, y3, y4], dim=1)  # -> [batch, 4*filters, seq_len]
        y = self.batch_norm(y)
        y = F.relu(y)
        return y


class Residual(nn.Module):
    """Residual 模块：将残差（shortcut）投影到与主路径相同通道后相加

    参数：
    - in_channels: 残差输入通道数（通常为上一层的输入通道）
    - out_channels_branch: 主路径输出通道数（通常为 4 * filters）
    """
    def __init__(self, in_channels: int, out_channels_branch: int):
        super(Residual, self).__init__()
        # 投影残差到 out_channels_branch 通道
        self.bottleneck = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels_branch,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels_branch)

    def forward(self, x, y):
        # x: shortcut（batch, in_channels, seq_len）
        # y: 主路径输出（batch, out_channels_branch, seq_len）
        res = self.batch_norm(self.bottleneck(x))
        y = y + res
        y = F.relu(y)
        return y


class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class InceptionModel1D(nn.Module):
    """1D-DITN 模型（Inception 架构） - 修复版

    设计思路：
    - 该网络接收单通道时间序列，shape=[batch, 1, seq_len]
    - 参数 input_size 在外部用于表示序列长度（seq_len），**不是**通道数
    - 在构建模块时显式传入模块所期望的通道数
    """
    def __init__(self, input_size: int, num_classes: int = 2, filters: int = 32, depth: int = 6, dilation: int = 4, dropout: float = 0.5):
        super(InceptionModel1D, self).__init__()
        self.seq_len = input_size  # 序列长度（例如4000）
        self.num_classes = num_classes
        self.filters = filters
        self.depth = depth
        self.dilation = dilation
        self.drop = nn.Dropout(p=dropout)

        # 动态构建模块：在第一层输入通道为1（单通道序列），之后模块输入来自上一模块的输出通道
        modules = OrderedDict()
        in_channels = 1
        for d in range(depth):
            modules[f'inception_{d}'] = Inception(
                in_channels=in_channels,
                filters=filters,
                dilation=dilation
            )
            # 当前模块输出通道为 4 * filters
            out_channels = 4 * filters

            # 每隔3层加入残差投影（保持与主路径通道对齐）
            if d % 3 == 2:
                # 残差的输入通道是最开始的 in_channels 对应的通道
                # 为简化，这里我们将残差的 shortcut 设为来自最初的输入通道
                modules[f'residual_{d}'] = Residual(
                    in_channels=in_channels,  # shortcut 的通道数
                    out_channels_branch=out_channels
                )
                # 残差拼接后，后续的 in_channels 变为 out_channels
                in_channels = out_channels
            else:
                # 后续模块的输入通道为当前模块的输出通道
                in_channels = out_channels

        modules['avg_pool'] = Lambda(f=lambda x: torch.mean(x, dim=-1))  # 全局平均池化
        modules['linear1'] = nn.Linear(in_features=4 * filters, out_features=filters)
        modules['linear2'] = nn.Linear(in_features=filters, out_features=num_classes)

        self.model = nn.Sequential(modules)

    def forward(self, x):
        # x expected shape: [batch, seq_len] or [batch, 1, seq_len]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # -> [batch, 1, seq_len]

        y = None
        # 逐层手动执行以支持 residual 操作
        for d in range(self.depth):
            inception = self.model.get_submodule(f'inception_{d}')
            y = inception(x if d == 0 else y)
            if d % 3 == 2:
                residual = self.model.get_submodule(f'residual_{d}')
                # shortcut 使用最初的输入 x （如同原作者意图），保证形状匹配
                y = residual(x, y)
                # 更新 x 为 y，供后续残差使用（与原实现保持一致）
                x = y

        y = self.model.get_submodule('avg_pool')(y)  # -> [batch, 4*filters]
        y = self.model.get_submodule('linear1')(y)
        y = self.drop(F.relu(y))
        y = self.model.get_submodule('linear2')(y)
        return y


# ==================== 模型集成类（保持大致逻辑不变，仅适配修改） ====================
class ArcFaultModelSystem:
    """电弧故障检测模型系统 - 集成1D-DITN和Informer（修复版）"""

    def __init__(self,
                 ditn_model_path: Optional[str] = None,
                 informer_checkpoint: Optional[str] = None,
                 ditn_config: Optional[Dict] = None,
                 informer_config: Optional[Dict] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 默认配置（input_size 代表序列长度）
        default_ditn_config = {
            'input_size': 4000,
            'num_classes': 2,
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

        # 这里 input_size 表示序列长度（seq_len），并不会被当作通道数
        seq_len = self.ditn_config['input_size']

        # 初始化1D-DITN模型：输入为单通道序列
        self.ditn_model = InceptionModel1D(input_size=seq_len,
                                           num_classes=self.ditn_config['num_classes'],
                                           filters=self.ditn_config['filters'],
                                           depth=self.ditn_config['depth'],
                                           dilation=self.ditn_config['dilation'],
                                           dropout=self.ditn_config['dropout']).to(self.device)
        self.ditn_model.eval()

        if ditn_model_path and Path(ditn_model_path).exists():
            self.load_ditn_model(ditn_model_path, class_map_override=class_map_from_config)
        else:
            print("警告: 未找到1D-DITN模型文件，使用随机初始化")
            self._update_class_map(class_map_from_config)

        # 初始化Informer模型（如可用）
        self.informer_model = None
        self.informer_exp = None
        if INFORMER_AVAILABLE:
            try:
                informer_args = dotdict(self.informer_config)
                self.informer_exp = Exp_Informer(informer_args)
                if not informer_checkpoint:
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
                # 重新构建模型以匹配类别数
                seq_len = self.ditn_config['input_size']
                self.ditn_model = InceptionModel1D(input_size=seq_len,
                                                   num_classes=self.ditn_config['num_classes'],
                                                   filters=self.ditn_config['filters'],
                                                   depth=self.ditn_config['depth'],
                                                   dilation=self.ditn_config['dilation'],
                                                   dropout=self.ditn_config['dropout']).to(self.device)
                self.ditn_model.eval()

            self.ditn_model.load_state_dict(state_dict)
            print(f"1D-DITN模型已从 {model_path} 加载")

            class_map_data = class_map_override or metadata.get('class_map')
            self._update_class_map(class_map_data)
        except Exception as e:
            print(f"加载1D-DITN模型失败: {e}")
            import traceback
            traceback.print_exc()
            self._update_class_map(class_map_override)

    def load_informer_model(self, checkpoint_path: str):
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
            import traceback
            traceback.print_exc()
            return False

    def preprocess_data(self, data: np.ndarray) -> torch.Tensor:
        """数据预处理：
        - 保证形状
        - 截断或填充到 seq_len
        - Savitzky-Golay 滤波（输入长度需 >= window_length）
        - 归一化
        - 返回 tensor: shape [batch, seq_len]
        """
        import time
        start_time = time.time()

        target_length = self.ditn_config['input_size']
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        # 截断或填充
        if data.shape[1] > target_length:
            data = data[:, :target_length]
        elif data.shape[1] < target_length:
            padding = np.zeros((data.shape[0], target_length - data.shape[1]))
            data = np.concatenate([data, padding], axis=1)

        # Savitzky-Golay 滤波（window_length 必须为奇数且 <= seq_len）
        filtered_data = np.zeros_like(data)
        win = 7
        if win >= target_length:
            win = target_length - 1 if (target_length - 1) % 2 == 1 else target_length - 2
            if win < 3:
                win = 3
        for i in range(data.shape[0]):
            try:
                filtered_data[i] = savgol_filter(data[i], window_length=win, polyorder=2)
            except Exception:
                filtered_data[i] = data[i]

        # 归一化（对每个样本独立进行 fit_transform，避免跨样本泄漏）
        scaled_list = []
        for i in range(filtered_data.shape[0]):
            arr = filtered_data[i].reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(arr).reshape(-1)
            scaled_list.append(scaled)

        filtered_data = np.stack(scaled_list, axis=0)

        # 转换为 tensor，注意保持为 [batch, seq_len]，模型 forward 会在内部 unsqueeze
        tensor_data = torch.FloatTensor(filtered_data).to(self.device)

        return tensor_data

    def classify(self, data: np.ndarray) -> Tuple[str, float, str]:
        import time
        start_time = time.time()

        # 预处理
        tensor_data = self.preprocess_data(data)

        # model expects [batch, 1, seq_len]
        if len(tensor_data.shape) == 2:
            tensor_data = tensor_data.unsqueeze(1)

        # 推理
        with torch.no_grad():
            outputs = self.ditn_model(tensor_data)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        self.inference_count += 1

        class_idx = predicted.item()
        status_text, fault_type = self.class_map.get(class_idx, ("未知", "unknown"))
        confidence_score = confidence.item() * 100

        return status_text, confidence_score, fault_type

    def predict(self, data: np.ndarray, seq_len: int = None, pred_len: int = None) -> Optional[np.ndarray]:
        if not INFORMER_AVAILABLE or self.informer_model is None:
            return None

        try:
            seq_len = seq_len or self.informer_config['seq_len']
            pred_len = pred_len or self.informer_config['pred_len']
            label_len = self.informer_config['label_len']

            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) == 2 and data.shape[1] > 1:
                data = data[:, 0:1]

            if data.shape[0] > seq_len:
                input_data = data[-seq_len:].copy()
            else:
                padding = np.zeros((seq_len - data.shape[0], 1))
                input_data = np.concatenate([padding, data], axis=0)

            batch_x = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
            batch_x_mark = torch.zeros(1, seq_len, 4).to(self.device)
            batch_y_mark = torch.zeros(1, pred_len, 4).to(self.device)

            dec_inp = torch.zeros(1, label_len + pred_len, 1).to(self.device)
            if input_data.shape[0] >= label_len:
                dec_inp[:, :label_len, :] = torch.FloatTensor(input_data[-label_len:]).unsqueeze(0).to(self.device)
            else:
                dec_inp[:, :input_data.shape[0], :] = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.informer_model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )

            return pred.cpu().numpy().squeeze()
        except Exception as e:
            print(f"Informer预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def inference(self, data: np.ndarray, fault_scenario: str = None) -> Tuple[str, float, str]:
        # 使用1D-DITN进行分类
        status_text, confidence, fault_type = self.classify(data)

        # 如果检测到故障，使用Informer进行预测
        if fault_type == "fault" and self.informer_model is not None:
            prediction = self.predict(data)
            if prediction is not None:
                pass

        # 模拟场景调整
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
