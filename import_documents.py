"""
批量导入知识库文档脚本

使用方法：
1. 将文档放在 knowledge_docs/ 目录下
2. 运行此脚本：python import_documents.py
"""

from knowledge_base import KnowledgeBase
import os
from pathlib import Path

def import_documents(docs_dir="knowledge_docs"):
    """批量导入文档到知识库"""
    
    # 初始化知识库
    print("正在初始化知识库...")
    kb = KnowledgeBase()
    
    # 检查目录是否存在
    if not os.path.exists(docs_dir):
        print(f"错误：目录 {docs_dir} 不存在")
        print(f"请创建目录并将文档放入其中")
        return
    
    # 收集所有支持的文件
    file_paths = []
    supported_extensions = ['.pdf', '.txt', '.md', '.docx', '.doc', '.csv']
    
    print(f"\n正在扫描 {docs_dir} 目录...")
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            if ext in supported_extensions:
                file_paths.append(file_path)
                print(f"  ✓ 找到: {file_path}")
    
    if not file_paths:
        print(f"\n未找到支持的文件格式")
        print(f"支持格式: {', '.join(supported_extensions)}")
        return
    
    print(f"\n找到 {len(file_paths)} 个文件，开始导入...")
    print("-" * 60)
    
    # 批量添加
    results = kb.add_documents(file_paths)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("导入完成！")
    print("=" * 60)
    print(f"✅ 成功: {len(results['success'])} 个文件")
    print(f"❌ 失败: {len(results['failed'])} 个文件")
    print(f"📄 总片段数: {results['total_chunks']}")
    
    if results['success']:
        print("\n成功导入的文件：")
        for item in results['success']:
            print(f"  ✓ {item['file']} ({item['chunks']} 片段)")
    
    if results['failed']:
        print("\n失败的文件：")
        for item in results['failed']:
            print(f"  ✗ {item['file']}: {item['reason']}")
    
    # 显示统计信息
    stats = kb.get_statistics()
    print("\n" + "=" * 60)
    print("知识库统计信息：")
    print("=" * 60)
    print(f"总文档片段: {stats['total_chunks']}")
    print(f"总文档数量: {stats['total_documents']}")
    print(f"总文件大小: {stats['total_size_mb']} MB")
    print(f"最后更新: {stats.get('last_updated', 'N/A')}")
    
    print("\n✅ 导入完成！")

if __name__ == "__main__":
    import_documents()

