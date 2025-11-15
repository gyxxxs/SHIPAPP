import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import json
from pathlib import Path
import time
from collections import Counter

# 尝试导入真实模型
try:
    from arc_models import ArcFaultModelSystem
    REAL_MODELS_AVAILABLE = True
except ImportError:
    REAL_MODELS_AVAILABLE = False
    print("警告: 真实模型模块未找到，将使用模拟模型")

class ArcFaultDetector(nn.Module):
    """电弧故障检测模型（1D CNN架构）"""
    
    def __init__(self, input_size=4000, hidden_size=128, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, seq_len]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ModelDiagnostics:
    """深度学习模型诊断系统"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 ditn_model_path: Optional[str] = None,
                 informer_checkpoint: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 尝试使用真实模型
        if REAL_MODELS_AVAILABLE:
            try:
                self.arc_model_system = ArcFaultModelSystem(
                    ditn_model_path=ditn_model_path,
                    informer_checkpoint=informer_checkpoint
                )
                self.use_real_models = True
                self.model = None  # 使用arc_model_system而不是单独的model
                print("✅ 已加载真实模型系统（1D-DITN + Informer）")
            except Exception as e:
                print(f"加载真实模型失败: {e}，使用模拟模型")
                self.use_real_models = False
                self.arc_model_system = None
                self.model = ArcFaultDetector().to(self.device)
                if model_path and Path(model_path).exists():
                    self.load_model(model_path)
                else:
                    self._initialize_model()
                self.model.eval()
        else:
            self.use_real_models = False
            self.arc_model_system = None
            self.model = ArcFaultDetector().to(self.device)
            if model_path and Path(model_path).exists():
                self.load_model(model_path)
            else:
                self._initialize_model()
            self.model.eval()
        
        # 诊断指标历史记录
        self.diagnostic_history: List[Dict] = []
        self.inference_times: List[float] = []
        self.confidence_scores: List[float] = []
        self.predictions: List[str] = []
        
        # 类别映射
        self.class_map = {
            0: ("运行正常 (安全)", "normal"),
            1: ("一级预警 (预测风险)", "early_arc"),
            2: ("二级预警 (故障确认)", "severe_arc"),
            3: ("干扰信号 (电机启动)", "motor_start")
        }
        
        # 模型统计
        self.total_inferences = 0
        self.start_time = time.time()
    
    def _initialize_model(self):
        """初始化模型权重"""
        for param in self.model.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, -0.1, 0.1)
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"模型已从 {model_path} 加载")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"模型已从 {model_path} 加载")
        except Exception as e:
            print(f"加载模型失败: {e}，使用随机初始化")
    
    def save_model(self, model_path: str):
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'class_map': self.class_map
            }, model_path)
            print(f"模型已保存到 {model_path}")
        except Exception as e:
            print(f"保存模型失败: {e}")
    
    def inference(self, data: np.ndarray, fault_scenario: str = None) -> Tuple[str, float, str]:
        """模型推理"""
        start_time = time.time()
        probabilities = None  # 初始化变量
        
        # 使用真实模型系统
        if self.use_real_models and self.arc_model_system:
            status_text, confidence_score, fault_type = self.arc_model_system.inference(data, fault_scenario)
            inference_time = (time.time() - start_time) * 1000
        else:
            # 使用模拟模型
            # 数据预处理
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            # 确保数据长度一致（截断或填充）
            target_length = 4000
            if data.shape[1] > target_length:
                data = data[:, :target_length]
            elif data.shape[1] < target_length:
                padding = np.zeros((data.shape[0], target_length - data.shape[1]))
                data = np.concatenate([data, padding], axis=1)
            
            # 归一化
            data_mean = np.mean(data)
            data_std = np.std(data) + 1e-8
            data = (data - data_mean) / data_std
            
            # 转换为tensor
            tensor_data = torch.FloatTensor(data).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(tensor_data)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 获取结果
            class_idx = predicted.item()
            status_text, fault_type = self.class_map.get(class_idx, ("未知", "unknown"))
            confidence_score = confidence.item() * 100
            
            # 如果是模拟模式，可以根据fault_scenario调整结果
            if fault_scenario:
                if fault_scenario == "severe_arc":
                    status_text, confidence_score, fault_type = "二级预警 (故障确认)", 97.5, "severe_arc"
                elif fault_scenario == "early_arc":
                    if confidence_score < 70:
                        status_text, confidence_score, fault_type = "一级预警 (预测风险)", min(90.0, confidence_score + 20), "early_arc"
                elif fault_scenario == "motor_start":
                    status_text, confidence_score, fault_type = "干扰信号 (电机启动)", 10.0, "motor_start"
                elif fault_scenario == "normal":
                    status_text, confidence_score, fault_type = "运行正常 (安全)", max(2.0, confidence_score), "normal"
        
        # 记录诊断信息
        self.inference_times.append(inference_time)
        self.confidence_scores.append(confidence_score)
        self.total_inferences += 1
        self.predictions.append(fault_type)
        
        # 记录详细诊断信息
        diagnostic_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": fault_type,
            "confidence": confidence_score,
            "inference_time_ms": round(inference_time, 2)
        }
        
        # 如果是模拟模型，添加概率信息
        if probabilities is not None:
            try:
                diagnostic_record["class_probabilities"] = {
                    k: round(v.item() * 100, 2) for k, v in enumerate(probabilities[0])
                }
            except:
                pass
        self.diagnostic_history.append(diagnostic_record)
        
        # 只保留最近1000条记录
        if len(self.diagnostic_history) > 1000:
            self.diagnostic_history = self.diagnostic_history[-1000:]
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]
        if len(self.confidence_scores) > 1000:
            self.confidence_scores = self.confidence_scores[-1000:]
        
        return status_text, confidence_score, fault_type
    
    def get_diagnostics(self) -> Dict:
        """获取模型诊断信息"""
        avg_inference_time = np.mean(self.inference_times[-100:]) if self.inference_times else 0
        avg_confidence = np.mean(self.confidence_scores[-100:]) if self.confidence_scores else 0
        max_inference_time = np.max(self.inference_times[-100:]) if self.inference_times else 0
        min_inference_time = np.min(self.inference_times[-100:]) if self.inference_times else 0
        
        uptime_hours = (time.time() - self.start_time) / 3600
        
        # 计算模型参数
        if self.use_real_models and self.arc_model_system:
            model_params = sum(p.numel() for p in self.arc_model_system.ditn_model.parameters())
            if self.arc_model_system.informer_model:
                model_params += sum(p.numel() for p in self.arc_model_system.informer_model.parameters())
            model_size_mb = round(model_params * 4 / (1024 * 1024), 2)
            model_info = "真实模型系统 (1D-DITN + Informer)"
        else:
            model_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
            model_size_mb = round(model_params * 4 / (1024 * 1024), 2) if self.model else 0
            model_info = "模拟模型"
        
        diagnostics = {
            "model_status": "运行中",
            "model_type": model_info,
            "device": str(self.device),
            "average_inference_time_ms": round(avg_inference_time, 2),
            "max_inference_time_ms": round(max_inference_time, 2),
            "min_inference_time_ms": round(min_inference_time, 2),
            "average_confidence": round(avg_confidence, 2),
            "total_inferences": self.total_inferences,
            "uptime_hours": round(uptime_hours, 2),
            "inferences_per_hour": round(self.total_inferences / max(uptime_hours, 0.01), 2),
            "recent_confidence_trend": self.confidence_scores[-10:] if len(self.confidence_scores) >= 10 else self.confidence_scores,
            "model_parameters": model_params,
            "model_size_mb": model_size_mb,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加真实模型统计信息
        if self.use_real_models and self.arc_model_system:
            real_stats = self.arc_model_system.get_statistics()
            diagnostics.update({
                "ditn_model": real_stats.get("ditn_model", "未知"),
                "informer_model": real_stats.get("informer_model", "未知")
            })
        
        # 性能评估
        if avg_inference_time > 50:
            diagnostics["performance_warning"] = "推理延迟较高，建议优化模型或使用GPU加速"
        elif avg_inference_time < 10:
            diagnostics["performance_status"] = "推理性能优秀"
        else:
            diagnostics["performance_status"] = "推理性能良好"
        
        if avg_confidence < 70:
            diagnostics["accuracy_warning"] = "平均置信度较低，建议重新训练模型或检查数据质量"
        elif avg_confidence > 90:
            diagnostics["accuracy_status"] = "模型置信度优秀"
        else:
            diagnostics["accuracy_status"] = "模型置信度良好"
        
        # 预测分布统计
        if self.predictions:
            pred_dist = Counter(self.predictions[-100:])
            diagnostics["recent_prediction_distribution"] = dict(pred_dist)
        
        return diagnostics
    
    def export_diagnostics_report(self, filepath: str):
        """导出诊断报告"""
        diagnostics = self.get_diagnostics()
        diagnostics["full_history"] = self.diagnostic_history[-100:]  # 只保存最近100条
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(diagnostics, f, ensure_ascii=False, indent=2)
        print(f"诊断报告已导出到 {filepath}")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.diagnostic_history = []
        self.inference_times = []
        self.confidence_scores = []
        self.predictions = []
        self.total_inferences = 0
        self.start_time = time.time()
        print("统计信息已重置")

