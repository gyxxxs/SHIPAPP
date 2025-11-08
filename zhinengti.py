import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import time
import os
import tempfile
from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import json

# 新增导入 - RAG和模型诊断
try:
    from knowledge_base import KnowledgeBase, init_knowledge_base
    from model_diagnostics import ModelDiagnostics
    RAG_AVAILABLE = True
    MODEL_DIAGNOSTICS_AVAILABLE = True
except ImportError as e:
    print(f"导入模块失败: {e}，将使用模拟模式")
    RAG_AVAILABLE = False
    MODEL_DIAGNOSTICS_AVAILABLE = False

# --- matplotlib 中文字体配置 ---
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 0. 环境和工具定义 (保持不变) ---

class ReportInput(BaseModel):
    """用于生成详细故障诊断报告的工具"""
    fault_id: str = Field(description="当前故障事件的唯一标识ID,例如:'EVENT-20251028-001'")
    severity: str = Field(description="故障的严重程度,例如:'一级预警'或'二级预警'")
    fault_type: str = Field(description="故障类型,如:'串联电弧故障'、'绝缘老化'等")

class StabilityInput(BaseModel):
    """用于查询船端边缘计算单元和船岸协同通信链路的实时状态和负载率"""

class MaintenanceInput(BaseModel):
    """根据故障类型生成维护工单"""
    circuit_id: str = Field(description="回路编号,例如:'03号舱回路'")
    fault_severity: str = Field(description="故障严重程度")
    maintenance_type: str = Field(description="维护类型:预防性/紧急")

def generate_diagnostic_report(fault_id: str, severity: str, fault_type: str) -> str:
    """生成格式化的故障诊断报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_data = {
        "report_id": f"RPT-{fault_id}",
        "timestamp": timestamp,
        "fault_severity": severity,
        "fault_type": fault_type,
        "dl_confidence": "97.5%",
        "root_cause": "高振动区域电缆固定件老化松动导致的串联电弧故障",
        "maintenance_advice": "立即进行预防性检查,紧固连接件,参考CCS规范第5.4.1条",
        "risk_level": "高" if "二级" in severity else "中"
    }
    return f"【诊断报告】{json.dumps(report_data, ensure_ascii=False, indent=2)}"

def check_system_stability() -> str:
    """查询系统稳定性状态"""
    stability_data = {
        "edge_compute_load": "38%",
        "inference_latency": "15ms",
        "communication_latency": "45ms",
        "model_accuracy": "97.5%",
        "system_status": "稳定"
    }
    return f"【系统状态】{json.dumps(stability_data, ensure_ascii=False)}"

def generate_maintenance_order(circuit_id: str, fault_severity: str, maintenance_type: str) -> str:
    """生成维护工单"""
    order_data = {
        "order_id": f"MO-{datetime.now().strftime('%Y%m%d%H%M')}",
        "circuit": circuit_id,
        "maintenance_type": maintenance_type,
        "priority": "紧急" if "二级" in fault_severity else "高",
        "required_tools": "红外热像仪,力矩扳手,绝缘测试仪",
        "estimated_duration": "2小时",
        "safety_requirements": "断电操作,穿戴PPE"
    }
    return f"【维护工单】{json.dumps(order_data, ensure_ascii=False)}"

AVAILABLE_TOOLS = {
    "generate_diagnostic_report": generate_diagnostic_report,
    "check_system_stability": check_system_stability,
    "generate_maintenance_order": generate_maintenance_order,
}

# --- 1. 增强的数据模拟 (保持不变) ---
def simulate_current_data(t, fault_scenario="normal", prediction_mode=False):
    """
    模拟更真实的船舶电流数据
    """
    base_frequency = 50
    # 模拟波形滚动，加入一个随机相位偏移
    phase_offset = time.time() * 2 * np.pi * base_frequency / 1000 
    
    time_series = np.linspace(0, 2 / base_frequency, t)  # 2个周期
    current = 10 * np.sin(2 * np.pi * base_frequency * time_series + phase_offset)
    
    # 基础噪声
    current += np.random.normal(0, 0.05, t)
    
    if fault_scenario == "early_arc":
        # 早期电弧特征:间歇性高频噪声
        mask = (time_series % 0.1 < 0.02)
        high_freq = np.sin(2 * np.pi * 5000 * time_series) * 0.3
        current += high_freq * mask
        
    elif fault_scenario == "severe_arc":
        # 严重电弧特征:持续高频噪声+幅值变化
        high_freq = np.sin(2 * np.pi * 3000 * time_series) * 0.8
        current += high_freq + 2 * np.random.rand(t)
        
    elif fault_scenario == "motor_start":
        # 电机启动干扰
        startup_effect = 3 * np.exp(-time_series * 2) * np.sin(2 * np.pi * 100 * time_series)
        current += startup_effect

    if prediction_mode:
        # 预测模式下的趋势特征
        trend_factor = (time.time() - st.session_state.last_update) / 10 
        trend = 0.5 * np.exp(-time_series * 3) * np.sin(2 * np.pi * 150 * time_series) * (1 + trend_factor)
        current += trend

    return time_series * 1000, current

# --- 2. 增强的模型推理 (集成真实模型) ---
@st.cache_resource
def get_model_diagnostics():
    """获取模型诊断实例"""
    if MODEL_DIAGNOSTICS_AVAILABLE:
        return ModelDiagnostics()
    return None

def dl_model_inference(data, fault_scenario):
    """使用真实深度学习模型进行推理"""
    model_diagnostics = get_model_diagnostics()
    
    if model_diagnostics is not None:
        # 使用真实模型推理
        status_text, confidence, fault_type = model_diagnostics.inference(data, fault_scenario)
        return status_text, confidence, fault_type
    else:
        # 回退到模拟模式
        if fault_scenario == "severe_arc":
            return "二级预警 (故障确认)", 97.5, "severe_arc"
        elif fault_scenario == "early_arc":
            if 'early_arc_confidence' not in st.session_state:
                 st.session_state.early_arc_confidence = 70.0
            
            st.session_state.early_arc_confidence = min(90.0, st.session_state.early_arc_confidence + 0.5) 

            if st.session_state.early_arc_confidence > 70.0:
                return "一级预警 (预测风险)", st.session_state.early_arc_confidence, "early_arc"
            else:
                return "运行正常 (安全)", 5.0, "normal"
                
        elif fault_scenario == "motor_start":
            return "干扰信号 (电机启动)", 10.0, "motor_start"
        else:
            st.session_state.early_arc_confidence = 70.0 if 'early_arc_confidence' in st.session_state else 70.0
            return "运行正常 (安全)", 2.0, "normal"

# --- 3. 智能体核心逻辑 (集成RAG) ---
@st.cache_resource
def get_gemini_client():
    """安全地获取 Gemini 客户端"""
    try:
        if "gemini_api_key" not in st.secrets:
            return None 
        GEMINI_API_KEY = st.secrets["gemini_api_key"]
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"初始化 Gemini 客户端失败: {e}")
        return None

@st.cache_resource
def get_knowledge_base():
    """获取知识库实例"""
    if RAG_AVAILABLE:
        return init_knowledge_base()
    return None

def gemini_agent_response(user_query: str, system_status: dict):
    """增强的智能体响应函数 - 集成RAG"""
    client = get_gemini_client()
    
    if client is None:
        return "⚠️ Gemini 客户端未初始化（可能缺少 API Key），无法执行 AI 推理。请检查配置。"
        
    status_context = (
        f"【实时系统状态】\n"
        f"- 检测状态: {system_status['detection_status']}\n"
        f"- 置信度: {system_status['confidence']:.1f}%\n" 
        f"- 故障类型: {system_status['fault_type']}\n"
        f"- 回路编号: {system_status['circuit_id']}\n"
        f"- 时间戳: {system_status['timestamp']}\n"
    )
    
    # 使用RAG检索（如果可用）
    kb = get_knowledge_base()
    if kb is not None:
        retrieval_results = kb.format_retrieval_results(user_query, k=5)
    else:
        # 回退到硬编码知识
        retrieval_results = (
        "【RAG检索结果:船舶电气安全知识库精要】\n"
        "--- 1. 预测与预警(基于 Informer 模型)---\n"
        " - **一级预警特征**:电流波形呈现不规则高频震荡(1-5kHz),幅值变化±15%,这是早期电弧的明确信号。\n"
        " - **二级预警特征**:持续高频噪声(3-8kHz),电流幅值异常波动超过±30%,需立即处理。\n"
        "--- 2. 故障诊断(历史经验归因)---\n"
        " - **根本原因**:80%的船舶电弧故障源于高振动区域的电缆连接点接触不良。\n"
        " - **典型位置**:机舱、货舱泵区、甲板机械供电回路。\n"
        "--- 3. 维护规范(船级社要求)---\n"
        " - **CCS规范第5.4.1条**:高振动区域每季度必须进行预防性检查和紧固维护。\n"
        " - **ABS规范第4-8-3条**:检测到电弧故障后,需在24小时内完成根本原因分析。\n"
    )

    system_instruction = (
        "你是一个专业的船舶电气安全智能体,基于船岸协同架构工作。"
        "你具备船舶电气安全的专业知识,同时也可以回答一般性问题。"
        "优先使用可用工具处理专业问题,对于工具无法处理的问题,请基于你的知识自主回答。"
        "回答要专业、准确、有帮助。"
    )
    
    full_prompt = (
        system_instruction + 
        "\n\n" + retrieval_results + 
        "\n\n" + status_context +  # 显式加入实时状态上下文
        "\n\n用户提问:" + user_query
    )

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=list(AVAILABLE_TOOLS.values()),
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=config,
        )
        
        if response.function_calls:
            function_call = response.function_calls[0]
            tool_name = function_call.name
            tool_args = dict(function_call.args)
            
            if tool_name in AVAILABLE_TOOLS:
                
                # 关键：确保工具参数使用最新的系统状态
                if tool_name == "generate_diagnostic_report":
                    tool_args['severity'] = system_status['detection_status']
                    tool_args['fault_type'] = system_status['fault_type']
                    tool_args['fault_id'] = f"EVENT-{datetime.now().strftime('%Y%m%d%H%M')}"
                elif tool_name == "generate_maintenance_order":
                    tool_args['fault_severity'] = system_status['detection_status']
                    tool_args['circuit_id'] = system_status['circuit_id']
                    tool_args['maintenance_type'] = "紧急" if "二级" in system_status['detection_status'] else "预防性"
                
                
                try:
                    tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
                    
                    response_after_tool = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            types.Content(role="user", parts=[types.Part.from_text(full_prompt)]),
                            types.Content(role="model", parts=[types.Part.from_function_call(function_call)]),
                            types.Content(role="tool", parts=[types.Part.from_text(tool_result)]),
                        ],
                        config=types.GenerateContentConfig(system_instruction=system_instruction),
                    )
                    return response_after_tool.text
                except Exception as tool_error:
                    st.warning(f"工具 {tool_name} 执行失败: {tool_error}")
                    pass 

        return response.text

    except Exception as e:
        error_msg = f"智能体 API 调用失败。错误信息: {e}"
        st.error(error_msg)
        
        fallback_responses = {
            "greeting": "您好!我是船舶电气安全助手。当前系统连接有些问题,但我能帮助分析故障预警、生成诊断报告和维护工单。",
            "status": f"当前监测状态:{system_status['detection_status']},置信度:{system_status['confidence']:.1f}%。由于系统暂时性问题,无法获取详细信息。",
            "general": "抱歉,当前系统暂时无法处理您的请求。请检查网络连接或稍后重试。对于船舶电气安全问题,通常建议检查电缆连接紧固性和绝缘状态。"
        }
        
        user_query_lower = user_query.lower()
        if any(word in user_query_lower for word in ['你好', '您好', 'hello', 'hi']):
            return fallback_responses['greeting']
        elif any(word in user_query_lower for word in ['状态', '检测', '预警', '故障']):
            return fallback_responses['status']
        else:
            return fallback_responses['general']

# --- 4. 主界面 ---
def main():
    st.set_page_config(layout="wide", page_title="船舶故障电弧智能监测与预警平台")
    st.title("🚢 船舶故障电弧智能监测与预警平台")
    st.markdown("**船岸协同架构 | 双重深度学习引擎 | 大模型智能体赋能**")

    # 初始化状态
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'fault_scenario' not in st.session_state:
        st.session_state.fault_scenario = "normal"
    if 'circuit_id' not in st.session_state:
        st.session_state.circuit_id = "03号舱回路"
    # --- 关键修改 1: 初始化 last_update ---
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()
    # --- 关键修改 2: 每次 Rerun 开始时更新 last_update ---
    # 这确保了无论 Rerun 是由用户交互还是定时器触发，时间基线都是最新的
    st.session_state.last_update = time.time() 
    
    if 'early_arc_confidence' not in st.session_state:
        st.session_state.early_arc_confidence = 70.0 

    get_gemini_client()

    # 侧边栏 - 系统配置 (保持不变)
    with st.sidebar:
        st.header("系统配置")
        st.session_state.circuit_id = st.selectbox(
            "监测回路",
            ["03号舱回路", "机舱主配电板", "货舱泵回路", "导航设备供电"]
        )
        
        st.subheader("故障场景模拟")
        scenario = st.radio(
            "选择运行模式:",
            ["正常运行", "早期电弧预警", "严重电弧故障", "电机启动干扰"]
        )
        
        scenario_map = {
            "正常运行": "normal",
            "早期电弧预警": "early_arc", 
            "严重电弧故障": "severe_arc",
            "电机启动干扰": "motor_start"
        }
        st.session_state.fault_scenario = scenario_map[scenario]
        
        st.subheader("系统信息")
        st.info("""
        **架构层级:**
        - 🚢 船端边缘计算
        - ☁️ 岸基智能体
        - 🔗 船岸协同
        """)
        
        # 新增：知识库管理
        st.subheader("📚 知识库管理")
        if RAG_AVAILABLE:
            kb = get_knowledge_base()
            if kb:
                stats = kb.get_statistics()
                doc_count = stats.get('total_chunks', 0)
                doc_num = stats.get('total_documents', 0)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("文档片段", doc_count)
                with col_b:
                    st.metric("文档数量", doc_num)
                
                # 知识库统计信息
                with st.expander("📊 详细统计"):
                    st.json(stats)
                
                # 文档列表
                with st.expander("📄 文档列表"):
                    documents = kb.list_documents()
                    if documents:
                        for doc in documents:
                            st.text(f"• {doc['name']} ({doc['chunks']} 片段, {doc.get('size', 0)/1024:.1f} KB)")
                    else:
                        st.info("暂无文档")
                
                # 添加文档
                with st.expander("➕ 添加文档"):
                    uploaded_files = st.file_uploader(
                        "上传文档 (PDF, TXT, MD, DOCX, CSV)",
                        type=['pdf', 'txt', 'md', 'docx', 'doc', 'csv'],
                        accept_multiple_files=True
                    )
                    if uploaded_files and st.button("添加到知识库"):
                        import tempfile
                        temp_paths = []
                        try:
                            for uploaded_file in uploaded_files:
                                # 保存到临时文件
                                temp_file = tempfile.NamedTemporaryFile(
                                    delete=False, 
                                    suffix=os.path.splitext(uploaded_file.name)[1]
                                )
                                temp_file.write(uploaded_file.getvalue())
                                temp_file.close()
                                temp_paths.append(temp_file.name)
                            
                            # 添加到知识库
                            with st.spinner("正在处理文档..."):
                                results = kb.add_documents(temp_paths)
                            
                            # 显示结果
                            if results['total_chunks'] > 0:
                                st.success(f"✅ 成功添加 {len(results['success'])} 个文档，共 {results['total_chunks']} 个片段")
                                if results['failed']:
                                    st.warning(f"⚠️ {len(results['failed'])} 个文档添加失败")
                            else:
                                st.error("❌ 所有文档添加失败")
                            
                            # 清理临时文件
                            for path in temp_paths:
                                try:
                                    os.unlink(path)
                                except:
                                    pass
                            
                            st.rerun()
                        except Exception as e:
                            st.error(f"添加文档失败: {e}")
                            # 清理临时文件
                            for path in temp_paths:
                                try:
                                    os.unlink(path)
                                except:
                                    pass
                
                # 知识库操作
                with st.expander("⚙️ 知识库操作"):
                    if st.button("🔄 刷新统计"):
                        st.rerun()
                    
                    if st.button("🗑️ 清空知识库", type="secondary"):
                        if st.session_state.get('confirm_clear', False):
                            if kb.clear_all():
                                st.success("知识库已清空")
                                st.session_state.confirm_clear = False
                                st.rerun()
                            else:
                                st.error("清空失败")
                        else:
                            st.session_state.confirm_clear = True
                            st.warning("⚠️ 再次点击确认清空（此操作不可恢复）")
            else:
                st.warning("📚 知识库: 未初始化")
        else:
            st.warning("📚 知识库: 不可用（模拟模式）")
        
        # 新增：模型诊断
        st.subheader("模型诊断")
        if MODEL_DIAGNOSTICS_AVAILABLE:
            model_diagnostics = get_model_diagnostics()
            if model_diagnostics:
                diagnostics = model_diagnostics.get_diagnostics()
                
                st.metric("总推理次数", diagnostics.get("total_inferences", 0))
                st.metric("平均延迟", f"{diagnostics.get('average_inference_time_ms', 0):.1f} ms")
                st.metric("平均置信度", f"{diagnostics.get('average_confidence', 0):.1f}%")
                
                if st.button("查看详细诊断报告"):
                    st.json(diagnostics)
                    
                    # 可视化诊断指标
                    if diagnostics.get("recent_confidence_trend"):
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # 置信度趋势
                        ax1.plot(diagnostics["recent_confidence_trend"])
                        ax1.set_title("最近10次推理置信度趋势")
                        ax1.set_xlabel("推理次数")
                        ax1.set_ylabel("置信度 (%)")
                        ax1.grid(True)
                        
                        # 预测分布
                        if diagnostics.get("recent_prediction_distribution"):
                            dist = diagnostics["recent_prediction_distribution"]
                            ax2.bar(dist.keys(), dist.values())
                            ax2.set_title("最近100次预测分布")
                            ax2.set_xlabel("故障类型")
                            ax2.set_ylabel("次数")
                            ax2.tick_params(axis='x', rotation=45)
                        
                        st.pyplot(fig)
                        plt.close(fig)
                
                if st.button("导出诊断报告"):
                    report_path = f"diagnostics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    model_diagnostics.export_diagnostics_report(report_path)
                    st.success(f"报告已导出: {report_path}")
                
                if st.button("重置统计"):
                    model_diagnostics.reset_statistics()
                    st.success("统计信息已重置")
                    st.rerun()
        else:
            st.warning("模型诊断: 不可用（模拟模式）")

    col1, col2 = st.columns([3, 2])

    # --- 实时监测 Dashboard ---
    with col1:
        st.header("📊 实时监测 Dashboard")
        
        t_series, current_data = simulate_current_data(
            t=4000, 
            fault_scenario=st.session_state.fault_scenario,
            prediction_mode=(st.session_state.fault_scenario == "early_arc")
        )
        
        # 模型推理
        status_text, confidence, fault_type = dl_model_inference(
            current_data, st.session_state.fault_scenario
        )
        
        # 系统状态
        system_status = {
            "detection_status": status_text,
            "confidence": confidence,
            "fault_type": fault_type,
            "circuit_id": st.session_state.circuit_id,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        # 状态颜色映射
        status_color = {
            "运行正常": "green",
            "干扰信号": "blue", 
            "一级预警": "orange",
            "二级预警": "red"
        }
        
        color = "green"
        for key, value in status_color.items():
            if key in status_text:
                color = value
                break
        
        # 1. 状态显示
        st.markdown(
            f"**检测状态:** <span style='color:{color}; font-size: 24px;'>{status_text}</span>",
            unsafe_allow_html=True
        )
        
        # 2. Metric
        st.metric("模型置信度", f"{confidence:.1f}%")
        st.metric("监测回路", st.session_state.circuit_id)


        # 3. 波形图
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_series, current_data, label=f'Current Waveform (A) @ {system_status["timestamp"]}', color=color, linewidth=1)
        ax.set_title(f" Real-time current waveform monitoring ")
        ax.set_xlabel("Time(ms)")
        ax.set_ylabel("Current(A)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-20, 20)
        ax.legend(loc='upper right')
        
        if st.session_state.fault_scenario == "early_arc":
            ax.plot(t_series, current_data + 2, label='Informer Predicted Risk Trend', color='purple', linestyle='--', alpha=0.7)
            ax.legend(loc='upper right')
        
        st.pyplot(fig)
        plt.close(fig)
        
        # 4. 预警/提示信息
        if "预警" in status_text:
            st.warning(f"🚨 **{status_text}** - 模型置信度 {confidence:.1f}%，请立即启动智能体进行诊断!")
        elif "干扰" in status_text:
            st.info("ℹ️ **干扰信号** - 检测到瞬时高频，判断为电机启动，请持续监测。")
        else:
            st.success("✅ **运行正常** - 系统稳定，故障率低。")

    # --- 智能体交互中心 (保持不变) ---
    with col2:
        st.header("💬 智能体交互中心")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        st.subheader("💡 预设问题")
        presets = {
            "前瞻预警": "当前波形走势是否正常?有无潜在的电弧风险?",
            "诊断查询": "请分析故障根本原因和船级社维护要求",
            "系统状态": "边缘计算单元和通信链路状态如何?",
            "维护指导": "根据当前预警生成维护工单"
        }
        
        preset_cols = st.columns(2)
        
        for i, (preset_name, preset_text) in enumerate(presets.items()):
            col = preset_cols[i % 2]
            if col.button(f"{preset_name}", key=preset_name):
                st.session_state.messages.append({"role": "user", "content": preset_text})
                
                with st.chat_message("user"):
                    st.markdown(preset_text)

                with st.chat_message("assistant"):
                    with st.spinner("智能体推理中..."):
                        response = gemini_agent_response(preset_text, system_status)
                    
                    full_response = ""
                    message_placeholder = st.empty()
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.01)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                        
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # 强制 Rerun 以确保界面和状态完全同步
                st.rerun()
        
        # 聊天输入
        if prompt := st.chat_input("请输入您的问题..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("智能体推理中..."):
                    response = gemini_agent_response(prompt, system_status)
                
                full_response = ""
                message_placeholder = st.empty()
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.01)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
            
    # --- 脚本末尾：定时刷新机制 (修复节流问题) ---
    
    # 我们知道脚本运行到这里用了不到 0.5s，所以 time.time() - last_update 应该小于 0.5。
    # 我们需要引入一个短暂的暂停，然后强制 Rerun，让下一次运行能看到最新的 last_update 时间。
    
    # 强制等待 0.5s - (当前运行时间)
    time_spent = time.time() - st.session_state.last_update 
    sleep_time = max(0, 0.5 - time_spent) # 确保至少暂停到 0.5s
    
    # 关键：如果用户在右侧进行了交互，脚本会在这里暂停一下，然后立即 Rerun。
    # 如果没有交互，脚本会等待直到 0.5s 满足，然后 Rerun。
    time.sleep(sleep_time)

    # 由于我们已经在开头更新了 last_update，这里直接强制 Rerun 即可实现连续循环
    st.rerun()


if __name__ == "__main__":
    main()
