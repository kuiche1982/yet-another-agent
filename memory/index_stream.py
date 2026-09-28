"""
流式索引工具 - 逐行处理，低内存占用
"""
import json
import sys
import os
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.retriever import BGERetriever
from memory.mem_manager import MemoryManager
import tempfile
import yaml

# 配置
raw_file = Path("data/raw/rollout-session.jsonl")
output_dir = Path("data/embeddings")

print(f"读取: {raw_file}")
print(f"输出: {output_dir}")

# 创建临时配置
config = {'use_llm_classifier': False}
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(config, f)
    config_path = f.name

try:
    # 初始化 MemoryManager
    mem_manager = MemoryManager(config_path=config_path, enable_auto_classify=False)

    # 初始化 BGERetriever（但先不清空，我们流式处理）
    retriever = BGERetriever()
    retriever.encode_documents([])  # 先清空

    # 统计
    total_count = 0
    type_stats = Counter()
    summarized_count = 0
    start_time = time.time()

    # 逐行处理
    print("开始流式处理...")
    with open(raw_file, 'r') as f:
        batch_texts = []
        batch_idx = []

        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                text = entry.get('text', '').strip()

                if not text:
                    continue

                total_count += 1

                # 简单分类（规则）
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

                type_stats[mem_type] += 1

                # 保存到 MemoryManager
                mem_manager.add_fragment(
                    text=text[:300],
                    type=mem_type,
                    project_id='rollout',
                    add_to_summary=False
                )

                batch_texts.append(text[:300])
                batch_idx.append(total_count)

                # 每10条编码并保存一次
                if len(batch_texts) >= 10:
                    retriever.encode_documents(batch_texts)
                    retriever._save_embeddings()
                    batch_texts = []
                    batch_idx = []

                    # 进度显示
                    if total_count % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"  处理进度: {total_count} 条, 用时: {elapsed:.1f}s, 类型: {dict(type_stats)}")

            except Exception as e:
                print(f"  跳过第 {line_num} 行: {e}")

    # 处理剩余的
    if batch_texts:
        retriever.encode_documents(batch_texts)
        retriever._save_embeddings()

    # 最终统计
    elapsed = time.time() - start_time
    print(f"\n处理完成！")
    print(f"总处理条数: {total_count}")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"类型分布: {dict(type_stats)}")
    print(f"\n检查 {output_dir}/dense:")
    if output_dir.exists():
        files = list(output_dir.glob("dense/*.npy"))
        print(f"  找到 {len(files)} 个文件")

    print("\n✅ 索引完成！")
    print(f"现在可以用 python memory/query.py --query '测试服务器ip' 测试查询")

finally:
    os.unlink(config_path)
