"""
简化版索引工具 - 不使用LLM，快速完成
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.retriever import BGERetriever
from memory.mem_manager import MemoryManager
import tempfile
import yaml
import os

# 读取原始数据
raw_file = Path("data/raw/rollout-session.jsonl")
output_dir = Path("data/embeddings")

print(f"读取: {raw_file}")

# 创建临时配置
config = {'use_llm_classifier': False}
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(config, f)
    config_path = f.name

try:
    # 初始化 MemoryManager
    mem_manager = MemoryManager(config_path=config_path, enable_auto_classify=False)

    # 读取数据并分类
    all_texts = []
    with open(raw_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            text = entry.get('text', '')

            # 简单分类：判断类型
            text_lower = text.lower()
            if any(kw in text_lower for kw in ['def ', 'class ', 'import ']):
                mem_type = 'code'
            elif any(kw in text_lower for kw in ['config', '环境变量', 'settings', '参数']):
                mem_type = 'config'
            elif any(kw in text_lower for kw in ['fact', 'ip', '地址', 'test']):
                mem_type = 'facts'
            elif any(kw in text_lower for kw in ['todo', '计划', '目标']):
                mem_type = 'task'
            else:
                mem_type = 'session'

            # 保存
            mem_manager.add_fragment(
                text=text[:300],  # 截断到300字符
                type=mem_type,
                project_id='rollout',
                add_to_summary=False
            )

            all_texts.append(text[:300])

    print(f"已保存 {len(all_texts)} 条记录")

    # 编码
    print("编码 embeddings...")
    retriever = BGERetriever()
    retriever.encode_documents(all_texts)

    print(f"编码完成！")
    print(f"Dense向量数: {len(retriever.dense_vectors)}")
    print(f"Sparse向量数: {len(retriever.sparse_vectors)}")

    # 手动保存 embeddings
    retriever._save_embeddings()

    # 检查输出目录
    print(f"\n检查 {output_dir}/dense:")
    if output_dir.exists():
        files = list(output_dir.glob("dense/*.npy"))
        print(f"  找到 {len(files)} 个文件")

    print("\n✅ 索引完成！")
    print(f"现在可以用 python memory/query.py --query '测试服务器ip' 测试查询")

finally:
    os.unlink(config_path)
