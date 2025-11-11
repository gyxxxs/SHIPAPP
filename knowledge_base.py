import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import json
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 兼容新版与旧版 LangChain 路径
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.document_loaders import (
        PyPDFLoader, TextLoader, UnstructuredMarkdownLoader,
        UnstructuredWordDocumentLoader, CSVLoader
    )
except ImportError:
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.document_loaders import (
        PyPDFLoader, TextLoader, UnstructuredMarkdownLoader,
        UnstructuredWordDocumentLoader, CSVLoader
    )
# 兼容新版与旧版 Document 导入
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.documents import Document
from collections import Counter

class KnowledgeBase:
    """船舶电气安全知识库管理类"""
    
    def __init__(self, persist_directory: str = "./knowledge_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # 使用中文嵌入模型
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            print(f"加载嵌入模型失败: {e}，使用默认模型")
            try:
                self.embeddings = HuggingFaceEmbeddings()
            except:
                raise Exception("无法加载嵌入模型，请检查依赖安装")
        
        # 初始化向量数据库
        try:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
            )
        except Exception as e:
            print(f"初始化向量数据库失败: {e}")
            raise
        
        # 文本分割器 - 优化参数
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 文档索引记录（用于管理）
        self.metadata_file = os.path.join(persist_directory, "metadata.json")
        self._load_metadata()
    
    def _load_metadata(self):
        """加载文档元数据"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.doc_metadata = json.load(f)
            except:
                self.doc_metadata = {}
        else:
            self.doc_metadata = {}
    
    def _save_metadata(self):
        """保存文档元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.doc_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存元数据失败: {e}")
    
    def _get_loader(self, file_path: str):
        """根据文件扩展名获取对应的加载器"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        loader_map = {
            '.pdf': PyPDFLoader,
            '.txt': lambda p: TextLoader(p, encoding='utf-8'),
            '.md': UnstructuredMarkdownLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.csv': CSVLoader,
        }
        
        loader_class = loader_map.get(file_ext)
        if loader_class:
            try:
                return loader_class(file_path)
            except Exception as e:
                print(f"创建加载器失败 {file_ext}: {e}")
                return None
        return None
    
    def add_documents(self, file_paths: List[str], metadata: Dict = None) -> Dict:
        """添加文档到知识库，返回详细结果"""
        results = {
            'success': [],
            'failed': [],
            'total_chunks': 0
        }
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                results['failed'].append({
                    'file': file_path,
                    'reason': '文件不存在'
                })
                continue
            
            try:
                loader = self._get_loader(file_path)
                if loader is None:
                    results['failed'].append({
                        'file': file_path,
                        'reason': '不支持的文件格式'
                    })
                    continue
                
                documents = loader.load()
                
                # 添加元数据
                file_basename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                
                for doc in documents:
                    doc.metadata['source'] = file_basename
                    doc.metadata['type'] = 'technical_doc'
                    doc.metadata['file_path'] = file_path
                    doc.metadata['file_size'] = file_size
                    doc.metadata['added_at'] = datetime.now().isoformat()
                    if metadata:
                        doc.metadata.update(metadata)
                
                # 分割文档
                texts = self.text_splitter.split_documents(documents)
                
                # 添加到向量数据库
                self.vectorstore.add_documents(texts)
                self.vectorstore.persist()
                
                # 记录元数据
                self.doc_metadata[file_basename] = {
                    'file_path': file_path,
                    'file_size': file_size,
                    'chunks': len(texts),
                    'added_at': datetime.now().isoformat(),
                    'modified_at': file_modified,
                    'type': 'file'
                }
                self._save_metadata()
                
                results['success'].append({
                    'file': file_path,
                    'chunks': len(texts)
                })
                results['total_chunks'] += len(texts)
                
            except Exception as e:
                results['failed'].append({
                    'file': file_path,
                    'reason': str(e)
                })
                continue
        
        return results
    
    def add_text(self, text: str, metadata: dict = None) -> int:
        """直接添加文本到知识库"""
        if metadata is None:
            metadata = {}
        
        metadata.setdefault('source', 'manual_input')
        metadata.setdefault('type', 'manual')
        metadata.setdefault('added_at', datetime.now().isoformat())
        
        doc = Document(page_content=text, metadata=metadata)
        texts = self.text_splitter.split_documents([doc])
        
        try:
            self.vectorstore.add_documents(texts)
            self.vectorstore.persist()
            
            # 记录元数据
            source_name = metadata.get('source', 'manual_input')
            self.doc_metadata[source_name] = {
                'chunks': len(texts),
                'added_at': metadata['added_at'],
                'type': 'manual'
            }
            self._save_metadata()
            
            return len(texts)
        except Exception as e:
            print(f"添加文本失败: {e}")
            return 0
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Document]:
        """检索相关文档"""
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"检索失败: {e}")
            return []
    
    def search_with_score(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[Tuple[Document, float]]:
        """检索文档并返回相似度分数（分数越低越相似）"""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            # 过滤低分（高相似度）结果
            filtered = [(doc, score) for doc, score in results if score <= score_threshold or score_threshold == 0.0]
            return filtered
        except Exception as e:
            print(f"检索失败: {e}")
            return []
    
    def search_mmr(self, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5) -> List[Document]:
        """使用MMR（最大边际相关性）重排序检索，提高结果多样性"""
        try:
            return self.vectorstore.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
        except Exception as e:
            print(f"MMR检索失败: {e}，回退到普通检索")
            return self.search(query, k=k)
    
    def format_retrieval_results(self, query: str, k: int = 5, use_mmr: bool = False, 
                                score_threshold: float = 0.0) -> str:
        """格式化检索结果为字符串，用于注入到prompt"""
        if use_mmr:
            results = self.search_mmr(query, k=k)
            scores = None
        else:
            results_with_scores = self.search_with_score(query, k=k, score_threshold=score_threshold)
            if results_with_scores:
                results = [doc for doc, _ in results_with_scores]
                scores = [score for _, score in results_with_scores]
            else:
                results = []
                scores = None
        
        if not results:
            return "【RAG检索结果:船舶电气安全知识库】\n未找到相关文档，将使用通用知识回答。\n"
        
        formatted = "【RAG检索结果:船舶电气安全知识库】\n"
        for i, doc in enumerate(results, 1):
            formatted += f"\n--- 检索片段 {i} ---\n"
            formatted += f"来源: {doc.metadata.get('source', '未知')}\n"
            if scores and i <= len(scores):
                # ChromaDB返回的距离，越小越相似，转换为相似度百分比
                similarity = max(0, min(100, (1 - scores[i-1]) * 100))
                formatted += f"相似度: {similarity:.1f}%\n"
            formatted += f"内容: {doc.page_content[:400]}...\n"  # 增加长度限制
        
        return formatted
    
    def get_collection_count(self) -> int:
        """获取知识库中的文档片段数量"""
        try:
            collection = self.vectorstore._collection
            return collection.count()
        except:
            return 0
    
    def get_statistics(self) -> Dict:
        """获取知识库详细统计信息"""
        stats = {
            'total_chunks': self.get_collection_count(),
            'total_documents': len(self.doc_metadata),
            'documents_by_type': Counter(),
            'documents_by_source': {},
            'total_size_bytes': 0,
            'last_updated': None
        }
        
        for doc_name, meta in self.doc_metadata.items():
            doc_type = meta.get('type', 'unknown')
            stats['documents_by_type'][doc_type] += 1
            
            source = meta.get('source', doc_name)
            if source not in stats['documents_by_source']:
                stats['documents_by_source'][source] = {
                    'chunks': meta.get('chunks', 0),
                    'size': meta.get('file_size', 0),
                    'added_at': meta.get('added_at', '')
                }
            
            stats['total_size_bytes'] += meta.get('file_size', 0)
            
            added_at = meta.get('added_at', '')
            if added_at:
                if stats['last_updated'] is None or added_at > stats['last_updated']:
                    stats['last_updated'] = added_at
        
        stats['documents_by_type'] = dict(stats['documents_by_type'])
        stats['total_size_mb'] = round(stats['total_size_bytes'] / (1024 * 1024), 2)
        
        return stats
    
    def list_documents(self) -> List[Dict]:
        """列出所有文档信息"""
        documents = []
        for doc_name, meta in self.doc_metadata.items():
            documents.append({
                'name': doc_name,
                'type': meta.get('type', 'unknown'),
                'chunks': meta.get('chunks', 0),
                'size': meta.get('file_size', 0),
                'added_at': meta.get('added_at', ''),
                'file_path': meta.get('file_path', '')
            })
        return documents
    
    def delete_document(self, doc_name: str) -> bool:
        """删除指定文档（通过源名称）"""
        # 注意：ChromaDB不直接支持按元数据删除，需要重建
        # 这里提供一个标记删除的方案
        if doc_name in self.doc_metadata:
            self.doc_metadata[doc_name]['deleted'] = True
            self.doc_metadata[doc_name]['deleted_at'] = datetime.now().isoformat()
            self._save_metadata()
            return True
        return False
    
    def search_by_source(self, source_name: str) -> List[Document]:
        """根据源名称检索文档"""
        try:
            # 使用filter进行元数据过滤
            results = self.vectorstore.similarity_search(
                "", k=1000  # 获取足够多的结果
            )
            # 手动过滤
            filtered = [doc for doc in results if doc.metadata.get('source') == source_name]
            return filtered
        except Exception as e:
            print(f"按源检索失败: {e}")
            return []
    
    def clear_all(self) -> bool:
        """清空知识库（危险操作）"""
        try:
            # 删除向量数据库
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            
            # 重新初始化
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            self.doc_metadata = {}
            self._save_metadata()
            return True
        except Exception as e:
            print(f"清空知识库失败: {e}")
            return False

def init_knowledge_base():
    """初始化知识库，添加默认知识文档"""
    kb = KnowledgeBase()
    
    # 检查是否已有数据
    if kb.get_collection_count() > 0:
        print("知识库已存在，跳过初始化")
        return kb
    
    # 创建扩展的默认知识文档（分多个部分添加，提高检索效果）
    knowledge_parts = [
        {
            'title': '预测与预警',
            'content': """
# 预测与预警（基于 Informer 模型）

## 一级预警特征
电流波形呈现不规则高频震荡(1-5kHz)，幅值变化±15%，这是早期电弧的明确信号。建议进行预防性检查，重点关注高振动区域的电缆连接点。一级预警通常在故障发生前24-48小时出现，是预防性维护的关键窗口期。一级预警时系统仍可正常运行，但需要密切监控。

## 二级预警特征
持续高频噪声(3-8kHz)，电流幅值异常波动超过±30%，需立即处理。可能引发火灾风险，必须立即断电检查。二级预警表示故障已经发生或即将发生，需要紧急响应。此时系统已处于危险状态，必须采取紧急措施。

## 预警响应流程
1. 一级预警：通知维护人员，安排预防性检查，记录波形数据，增加监测频率
2. 二级预警：立即断电，启动应急响应程序，隔离故障回路，通知岸基支持

## 预警阈值设置
- 一级预警阈值：高频分量>5%，幅值变化>10%，持续时间>10分钟
- 二级预警阈值：高频分量>15%，幅值变化>25%，持续时间>5分钟
- 紧急阈值：高频分量>30%，幅值变化>50%，持续时间>1分钟

## 预警信号识别
- 时域特征：电流波形出现尖峰、毛刺、不规则震荡
- 频域特征：出现1-10kHz的高频分量，谐波成分异常
- 统计特征：电流有效值、峰值、均方根值异常波动
- 趋势特征：电流幅值持续上升或下降，变化率超过阈值

## 预警准确性提升
- 多传感器融合：结合电流、电压、温度、振动等多源数据
- 历史数据对比：与正常运行状态和历史故障案例对比
- 环境因素考虑：考虑船舶运行状态、环境温度、湿度等
- 模型持续优化：根据实际运行数据不断优化预警模型
"""
        },
        {
            'title': '故障诊断',
            'content': """
# 故障诊断（历史经验归因）

## 根本原因分析
根据历史数据分析，80%的船舶电弧故障源于高振动区域的电缆连接点接触不良。15%源于绝缘老化，5%源于设备过载。其他原因包括：环境腐蚀(3%)、设计缺陷(2%)。接触不良的主要原因包括：连接件松动、氧化腐蚀、热胀冷缩、机械振动。

## 典型故障位置
- 机舱主配电板：振动大，温度高，故障率最高(35%)，平均故障间隔时间(MTBF)约2年
- 货舱泵区：潮湿环境，腐蚀风险，故障率(25%)，MTBF约3年
- 甲板机械供电回路：暴露在恶劣环境中，故障率(20%)，MTBF约4年
- 导航设备供电系统：对稳定性要求极高，故障率(10%)，MTBF约5年
- 其他区域：故障率(10%)，MTBF约6年

## 故障诊断步骤
1. 检查电流波形特征：分析频率成分、幅值变化、相位关系
2. 定位故障回路：使用回路识别算法、阻抗测量、信号注入
3. 分析历史维护记录：查看该回路的维护历史、故障记录
4. 确定根本原因：结合现场检查结果、环境因素、运行工况
5. 制定维护方案：根据故障类型和严重程度，制定详细维护计划

## 故障分类
- 串联电弧：电流路径中断，产生高频噪声(1-10kHz)，电流幅值下降
- 并联电弧：电流路径短路，产生大电流冲击，可能触发保护装置
- 接地故障：对地短路，产生漏电流，可能导致人员触电
- 绝缘老化：绝缘电阻下降，温升异常，可能导致击穿
- 接触不良：连接点电阻增大，产生局部过热，可能导致火灾
- 过载故障：电流超过额定值，设备过热，可能损坏设备

## 故障诊断工具
- 示波器：观察电流波形，分析时域特征
- 频谱分析仪：分析频率成分，识别故障特征频率
- 绝缘测试仪：测量绝缘电阻，判断绝缘状态
- 热像仪：检测过热点，定位故障位置
- 回路识别器：识别故障回路，快速定位

## 故障诊断技巧
- 对比分析法：与正常运行状态对比，找出异常
- 历史数据法：参考历史故障案例，快速诊断
- 分段测试法：分段测试，逐步缩小故障范围
- 环境因素法：考虑环境温度、湿度、振动等影响
- 多参数综合法：综合电流、电压、温度、振动等多参数
"""
        },
        {
            'title': '维护规范',
            'content': """
# 维护规范（船级社要求）

## CCS规范第5.4.1条
高振动区域每季度必须进行预防性检查和紧固维护。检查项目包括：连接件紧固度、绝缘电阻测试、温升情况监测、接地系统检查。检查结果必须记录在维护日志中，并保存至少3年。对于关键设备，检查周期应缩短至每月一次。

## ABS规范第4-8-3条
检测到电弧故障后，需在24小时内完成根本原因分析。必须详细记录故障类型、位置、处理措施和预防方案。所有故障报告必须提交给船级社审核，并在30天内完成整改。对于重大故障，必须在48小时内报告。

## IMO规范要求
所有电气设备必须符合SOLAS公约要求。定期进行绝缘测试和接地检查，确保船舶电气系统安全可靠。绝缘电阻测试应每月进行一次，接地系统检查应每季度进行一次。所有测试结果必须符合IMO最低标准。

## DNV规范要求
DNV规范要求所有电气设备必须定期进行预防性维护。对于关键系统，维护周期不得超过3个月。所有维护活动必须记录在维护管理系统中，并定期审核。

## LR规范要求
劳氏船级社要求建立完善的维护管理体系。所有维护活动必须按照维护计划执行，不得随意更改。对于发现的缺陷，必须及时处理，不得带病运行。

## 维护工具清单
- 红外热像仪：检测过热点，温度范围-20°C至350°C，精度±2°C
- 力矩扳手：紧固连接件，扭矩范围5-100N·m，精度±3%
- 绝缘测试仪：测量绝缘电阻，测试电压500V/1000V，量程0.1MΩ-100GΩ
- 接地电阻测试仪：检查接地系统，测试范围0.01Ω-2000Ω，精度±2%
- 示波器：分析电流波形，采样率>10kHz，带宽>100kHz
- 频谱分析仪：分析频率成分，频率范围0-100kHz，分辨率1Hz
- 万用表：测量电压、电流、电阻，精度±0.5%
- 钳形电流表：测量交流电流，量程0-1000A，精度±2%

## 维护周期
- 日常检查：每日一次，检查运行状态、指示灯、仪表读数
- 周检：每周一次，检查连接紧固度、外观检查、清洁保养
- 月检：每月一次，绝缘电阻测试、接地系统检查、功能测试
- 季检：每季度一次，全面预防性维护、紧固检查、更换易损件
- 年检：每年一次，系统全面检查、性能测试、校准仪器

## 维护记录要求
- 维护日期和时间
- 维护人员姓名和资质
- 维护内容和结果
- 发现的问题和处理措施
- 使用的工具和材料
- 下次维护计划
- 维护负责人签字
"""
        },
        {
            'title': '系统架构',
            'content': """
# 系统架构说明

## 船岸协同架构
- 船端边缘计算：实时数据采集和初步分析，延迟<5ms
- 岸基智能体：深度分析和决策支持，延迟<50ms
- 通信链路：确保数据实时传输，带宽>10Mbps

## 双重深度学习引擎
- Informer模型：用于时间序列预测和早期预警，准确率>92%
- CNN模型：用于故障模式识别和分类，准确率>95%

## 性能指标
- 边缘计算负载率：通常保持在30-40%，峰值<70%
- 推理延迟：目标<20ms，实际<15ms
- 通信延迟：目标<50ms，实际<45ms
- 模型准确率：目标>95%，实际>97%

## 数据采集
- 采样频率：10kHz，满足奈奎斯特定理
- 数据长度：每个样本4000个点，覆盖2个工频周期
- 数据存储：本地存储7天，云端存储90天
"""
        },
        {
            'title': '安全操作规程',
            'content': """
# 安全操作规程

## 故障处理安全要求
1. 发现二级预警时，必须立即断电，禁止带电操作
2. 进行维护作业前，必须确认回路已断电并挂牌
3. 使用绝缘工具，穿戴个人防护装备(PPE)
4. 两人以上作业，一人操作，一人监护
5. 作业完成后，检查无误方可恢复供电

## 应急响应程序
1. 立即断电：切断故障回路电源
2. 隔离故障：断开相关开关和断路器
3. 通知相关人员：船长、轮机长、岸基支持
4. 现场检查：确认故障位置和原因
5. 制定修复方案：根据故障类型制定方案
6. 执行修复：按照安全规程执行
7. 测试验证：修复后进行全面测试
8. 恢复供电：确认安全后恢复供电
9. 记录归档：详细记录故障和处理过程

## 个人防护装备
- 绝缘手套：耐压等级>500V
- 绝缘鞋：耐压等级>500V
- 安全帽：符合GB2811标准
- 防护眼镜：防冲击、防电弧
- 工作服：阻燃材料，防静电
"""
        },
        {
            'title': '常见问题解答',
            'content': """
# 常见问题解答

## Q1: 如何区分一级预警和二级预警？
A: 一级预警的特征是间歇性高频噪声，幅值变化较小(±15%)；二级预警的特征是持续高频噪声，幅值变化较大(±30%)。一级预警可以继续观察，二级预警必须立即处理。可以通过波形持续时间、频率成分、幅值变化率等参数综合判断。

## Q2: 预警误报率高怎么办？
A: 可以通过调整预警阈值、增加历史数据训练、使用多模型融合等方式降低误报率。建议定期校准模型参数，使用交叉验证方法评估模型性能。对于特定环境，可以建立环境特定的预警模型。

## Q3: 如何处理历史数据不足的情况？
A: 可以使用数据增强技术、迁移学习、或者使用模拟数据补充。同时建议建立数据采集规范，逐步积累数据。可以使用其他船舶或类似系统的数据进行迁移学习。

## Q4: 维护周期是否可以延长？
A: 维护周期应根据实际情况调整，但不得低于船级社最低要求。对于高故障率区域，应缩短维护周期。可以通过可靠性分析、故障率统计等方法优化维护周期。

## Q5: 如何提高系统可靠性？
A: 可以从以下几个方面提高：1) 使用冗余设计 2) 定期校准和维护 3) 建立完善的监控体系 4) 加强人员培训 5) 使用高质量设备 6) 优化系统设计

## Q6: 绝缘电阻测试的标准是什么？
A: 根据IMO规范，新设备绝缘电阻应≥1MΩ，运行中设备应≥0.5MΩ。对于不同电压等级，标准有所不同。测试时应使用合适的测试电压，通常为设备额定电压的1.5-2倍。

## Q7: 如何判断连接件是否需要紧固？
A: 可以通过以下方法判断：1) 检查连接件是否有松动迹象 2) 测量连接点电阻，正常应<0.1Ω 3) 使用热像仪检查是否有过热点 4) 检查连接件是否有氧化腐蚀

## Q8: 故障发生后如何处理？
A: 故障处理流程：1) 立即断电，确保安全 2) 隔离故障回路 3) 通知相关人员 4) 现场检查，确定故障原因 5) 制定修复方案 6) 执行修复 7) 测试验证 8) 恢复供电 9) 记录归档

## Q9: 如何预防电弧故障？
A: 预防措施包括：1) 定期检查和紧固连接件 2) 保持设备清洁干燥 3) 定期进行绝缘测试 4) 使用高质量连接件 5) 避免过载运行 6) 及时更换老化设备 7) 加强环境控制

## Q10: 系统维护成本如何控制？
A: 可以通过以下方式控制成本：1) 优化维护周期，避免过度维护 2) 使用预防性维护，减少故障维修 3) 建立备件库存，减少停机时间 4) 培训维护人员，提高维护效率 5) 使用维护管理软件，提高管理效率
"""
        },
        {
            'title': '电气设备维护',
            'content': """
# 电气设备维护要点

## 主配电板维护
- 每日检查：指示灯、仪表读数、运行声音
- 每周检查：连接紧固度、外观清洁、通风情况
- 每月检查：绝缘电阻、接地系统、保护装置功能
- 每季度检查：全面紧固、更换易损件、性能测试
- 注意事项：保持干燥清洁，避免过载，定期校准仪表

## 电机维护
- 日常检查：运行声音、振动、温度、电流
- 周检：清洁、润滑、检查接线
- 月检：绝缘测试、轴承检查、性能测试
- 季检：全面解体检查、更换易损件
- 关键参数：绝缘电阻>1MΩ，轴承温度<70°C，振动<4.5mm/s

## 电缆维护
- 定期检查：外观检查、绝缘测试、连接检查
- 环境控制：保持干燥、避免机械损伤、防止腐蚀
- 更换标准：绝缘电阻<0.5MΩ，外观严重损坏，超过使用寿命
- 安装要求：避免过度弯曲、保持适当间距、正确接地

## 开关设备维护
- 定期检查：操作功能、接触电阻、绝缘状态
- 清洁保养：清除灰尘、检查触头、润滑机构
- 功能测试：操作试验、保护功能测试、联锁试验
- 更换标准：操作不灵活、接触不良、绝缘损坏

## 保护装置维护
- 定期测试：保护功能、动作值、动作时间
- 校准要求：每年校准一次，精度±2%
- 功能检查：过流保护、接地保护、差动保护
- 记录要求：测试结果、校准证书、维护记录
"""
        },
        {
            'title': '环境因素影响',
            'content': """
# 环境因素对电气系统的影响

## 温度影响
- 高温影响：加速绝缘老化、降低载流量、增加接触电阻
- 低温影响：材料变脆、润滑不良、启动困难
- 控制措施：保持适当温度(0-40°C)、加强通风、使用耐温材料
- 监测要求：关键设备温度监测，报警温度设置

## 湿度影响
- 高湿度影响：降低绝缘电阻、加速腐蚀、引起凝露
- 控制措施：保持相对湿度<85%、使用除湿设备、密封保护
- 监测要求：湿度监测，超过阈值报警
- 防护措施：使用防潮材料、加强密封、定期除湿

## 振动影响
- 振动来源：主机振动、海浪冲击、设备运行
- 影响：连接松动、机械疲劳、绝缘损坏
- 控制措施：减振设计、定期紧固、使用防振材料
- 监测要求：关键设备振动监测，超过阈值报警

## 腐蚀影响
- 腐蚀类型：电化学腐蚀、化学腐蚀、生物腐蚀
- 影响因素：湿度、盐雾、温度、污染物
- 防护措施：使用防腐材料、表面处理、定期清洁
- 检查要求：定期外观检查，发现腐蚀及时处理

## 盐雾影响
- 影响：加速腐蚀、降低绝缘性能、影响接触
- 防护措施：密封保护、使用耐盐雾材料、定期清洁
- 检查周期：每月检查一次，发现问题及时处理
"""
        },
        {
            'title': '故障案例分析',
            'content': """
# 典型故障案例分析

## 案例1：机舱主配电板连接松动
- 故障现象：一级预警，电流波形异常，连接点过热
- 故障原因：长期振动导致连接件松动，接触电阻增大
- 处理措施：断电、紧固连接件、清洁接触面、测试验证
- 预防措施：每季度紧固检查，使用防松垫片，加强减振

## 案例2：货舱泵区绝缘老化
- 故障现象：绝缘电阻下降，漏电流增大，二级预警
- 故障原因：高湿度环境，绝缘材料老化，局部击穿
- 处理措施：更换电缆，加强密封，改善环境
- 预防措施：定期绝缘测试，加强环境控制，及时更换老化设备

## 案例3：甲板机械供电回路过载
- 故障现象：电流超过额定值，设备过热，保护动作
- 故障原因：负载增加，设备容量不足，设计不合理
- 处理措施：减少负载，更换大容量设备，优化设计
- 预防措施：合理设计容量，避免过载运行，加强监测

## 案例4：导航设备供电系统干扰
- 故障现象：信号干扰，设备误动作，系统不稳定
- 故障原因：电磁干扰，接地不良，屏蔽不足
- 处理措施：改善接地，加强屏蔽，使用滤波器
- 预防措施：合理布线，加强屏蔽，改善接地系统

## 案例5：串联电弧故障
- 故障现象：高频噪声，电流下降，局部过热
- 故障原因：连接点接触不良，产生电弧
- 处理措施：断电检查，修复连接，更换损坏部件
- 预防措施：定期检查连接，使用高质量连接件，及时处理异常
"""
        },
        {
            'title': '技术参数标准',
            'content': """
# 技术参数和标准

## 绝缘电阻标准
- 新设备：≥1MΩ (500V测试电压)
- 运行设备：≥0.5MΩ (500V测试电压)
- 高压设备：≥10MΩ (1000V测试电压)
- 测试方法：使用绝缘测试仪，测试时间1分钟
- 记录要求：记录测试值、测试电压、测试时间、环境条件

## 接地电阻标准
- 系统接地：≤4Ω (对于380V系统)
- 设备接地：≤10Ω (对于一般设备)
- 防雷接地：≤1Ω (对于防雷系统)
- 测试方法：使用接地电阻测试仪，三极法测试
- 记录要求：记录测试值、测试方法、测试位置

## 电流参数
- 额定电流：设备铭牌标注的额定值
- 过载能力：短时过载1.5倍，持续1小时
- 启动电流：电机启动电流为额定电流的5-7倍
- 测量精度：±2% (使用标准电流表)
- 监测要求：关键设备实时监测，超过阈值报警

## 电压参数
- 额定电压：380V/220V (三相/单相)
- 电压波动：±5% (正常运行范围)
- 电压不平衡：≤2% (三相系统)
- 测量精度：±0.5% (使用标准电压表)
- 监测要求：实时监测，超过范围报警

## 温度参数
- 环境温度：0-40°C (正常运行范围)
- 设备温度：≤70°C (一般设备)
- 连接点温度：≤90°C (关键连接点)
- 测量方法：使用热像仪或温度计
- 监测要求：关键设备温度监测，超过阈值报警

## 振动参数
- 一般设备：≤4.5mm/s (振动速度)
- 精密设备：≤2.8mm/s (振动速度)
- 测量方法：使用振动测试仪
- 监测要求：关键设备振动监测，超过阈值报警
"""
        },
        {
            'title': '预防性维护策略',
            'content': """
# 预防性维护策略

## 基于时间的维护(TBM)
- 适用场景：故障模式与时间相关，有明确的寿命周期
- 维护周期：根据设备寿命和故障率确定
- 优点：简单易行，便于管理
- 缺点：可能过度维护或维护不足
- 应用：定期更换易损件，定期检查

## 基于状态的维护(CBM)
- 适用场景：可以监测设备状态，有明确的故障征兆
- 监测参数：温度、振动、电流、绝缘等
- 优点：按需维护，减少停机时间
- 缺点：需要监测设备，成本较高
- 应用：关键设备状态监测，预警维护

## 基于可靠性的维护(RBM)
- 适用场景：有历史故障数据，可以分析故障规律
- 分析方法：故障树分析，可靠性分析
- 优点：科学合理，优化维护资源
- 缺点：需要大量数据，分析复杂
- 应用：优化维护周期，制定维护计划

## 维护策略选择
- 关键设备：使用CBM，实时监测，及时维护
- 一般设备：使用TBM，定期维护，降低成本
- 非关键设备：使用RBM，优化维护，提高效率
- 组合策略：根据设备特点，组合使用多种策略

## 维护计划制定
- 设备分类：按重要性、故障率、维护难度分类
- 维护周期：根据设备类型和维护策略确定
- 资源分配：合理分配人力、物力、时间
- 计划执行：严格按照计划执行，不得随意更改
- 效果评估：定期评估维护效果，优化维护计划
"""
        },
        {
            'title': '应急处理预案',
            'content': """
# 应急处理预案

## 一级预警应急处理
- 立即行动：通知维护人员，增加监测频率，准备维护工具
- 检查内容：检查相关回路，分析波形数据，查找异常原因
- 处理措施：根据检查结果，采取相应措施，记录处理过程
- 时间要求：24小时内完成检查和处理
- 记录要求：详细记录预警信息、检查结果、处理措施

## 二级预警应急处理
- 立即行动：立即断电，隔离故障回路，通知相关人员
- 安全措施：确保人员安全，防止事故扩大，设置警戒区域
- 检查内容：现场检查，确定故障位置和原因
- 处理措施：制定修复方案，执行修复，测试验证
- 时间要求：4小时内完成初步处理，24小时内完成修复
- 记录要求：详细记录故障信息、处理过程、修复结果

## 火灾应急处理
- 立即行动：切断电源，使用灭火器，通知消防部门
- 安全措施：确保人员安全，防止火势蔓延，疏散人员
- 灭火方法：使用干粉灭火器或二氧化碳灭火器
- 注意事项：不得使用水灭火，防止触电，注意通风
- 后续处理：火灾扑灭后，检查设备，评估损失，制定修复方案

## 人员触电应急处理
- 立即行动：切断电源，使用绝缘工具脱离电源，进行急救
- 安全措施：确保施救人员安全，使用绝缘工具，防止二次伤害
- 急救方法：进行心肺复苏，及时送医，保持呼吸道通畅
- 注意事项：不得直接接触触电人员，确保自身安全
- 后续处理：调查事故原因，制定预防措施，加强安全培训

## 应急物资准备
- 灭火器：干粉灭火器、二氧化碳灭火器，定期检查
- 急救用品：急救箱、担架、氧气瓶，定期检查
- 绝缘工具：绝缘手套、绝缘鞋、绝缘垫，定期测试
- 应急照明：应急灯、手电筒，定期充电
- 通信设备：对讲机、电话，保持畅通
"""
        },
        {
            'title': '设备选型和配置',
            'content': """
# 设备选型和配置

## 电缆选型
- 电压等级：根据系统电压选择，留有余量
- 载流量：根据负载电流选择，考虑环境温度、敷设方式
- 绝缘材料：根据环境条件选择，考虑温度、湿度、腐蚀
- 防护等级：根据安装位置选择，IP等级满足要求
- 标准要求：符合IEC、GB等标准要求

## 开关选型
- 额定电流：根据负载电流选择，留有余量(1.25-1.5倍)
- 分断能力：根据短路电流选择，满足保护要求
- 操作方式：根据控制要求选择，手动或电动
- 保护功能：根据保护要求选择，过流、接地、差动等
- 环境适应：根据环境条件选择，温度、湿度、防护等级

## 保护装置选型
- 保护类型：过流保护、接地保护、差动保护等
- 动作值：根据保护要求设置，考虑选择性
- 动作时间：根据保护要求设置，满足快速性
- 精度要求：根据保护要求选择，一般±2%
- 可靠性：选择高可靠性产品，满足长期运行要求

## 监测设备选型
- 测量精度：根据监测要求选择，一般±1%
- 采样频率：根据信号特征选择，满足奈奎斯特定理
- 通信接口：根据系统要求选择，支持标准协议
- 环境适应：根据安装环境选择，满足环境要求
- 可靠性：选择高可靠性产品，满足长期运行要求

## 系统配置原则
- 可靠性：关键系统冗余配置，提高可靠性
- 经济性：合理配置，避免过度配置
- 可维护性：便于维护，易于更换
- 可扩展性：留有扩展空间，便于升级
- 标准化：使用标准产品，便于采购和维护
"""
        },
        {
            'title': '测试和验证方法',
            'content': """
# 测试和验证方法

## 绝缘电阻测试
- 测试设备：绝缘测试仪，测试电压500V/1000V
- 测试方法：断开电源，连接测试仪，测试1分钟
- 测试标准：新设备≥1MΩ，运行设备≥0.5MΩ
- 注意事项：测试前放电，测试后放电，记录环境条件
- 记录要求：记录测试值、测试电压、测试时间、环境条件

## 接地电阻测试
- 测试设备：接地电阻测试仪，三极法测试
- 测试方法：布置测试电极，连接测试仪，读取测试值
- 测试标准：系统接地≤4Ω，设备接地≤10Ω
- 注意事项：测试环境干燥，电极布置正确，避免干扰
- 记录要求：记录测试值、测试方法、测试位置、环境条件

## 功能测试
- 测试内容：操作功能、保护功能、联锁功能
- 测试方法：模拟操作，检查动作，验证功能
- 测试标准：功能正常，动作准确，时间满足要求
- 注意事项：安全操作，防止误动作，记录测试结果
- 记录要求：记录测试内容、测试结果、测试人员、测试时间

## 性能测试
- 测试内容：电压、电流、功率、效率等
- 测试方法：使用标准仪表，测量参数，计算性能
- 测试标准：满足设计要求，符合标准规定
- 注意事项：使用标准仪表，正确接线，记录环境条件
- 记录要求：记录测试参数、测试结果、测试条件、测试人员

## 验收测试
- 测试内容：全面功能测试、性能测试、安全测试
- 测试方法：按照验收标准，逐项测试，记录结果
- 测试标准：满足设计要求，符合标准规定
- 注意事项：严格按标准执行，不得降低要求
- 记录要求：详细记录测试内容、测试结果、验收结论
"""
        }
    ]
    
    # 分别添加各个知识部分
    for part in knowledge_parts:
        kb.add_text(
            part['content'], 
            metadata={
                'source': f'default_knowledge_{part["title"]}',
                'type': 'manual',
                'category': part['title']
            }
        )
    
    print("知识库初始化完成，已添加扩展知识内容")
    return kb

