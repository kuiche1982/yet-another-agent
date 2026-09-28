"""
数据导入管道 - 从 session 文件导入数据到记忆系统
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from .mem_manager import MemoryManager
from .incremental_tree import FragmentNode


class SessionPipeline:
    """Session 数据导入管道"""

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager

    def import_session(
        self,
        session_path: str,
        project_id: Optional[str] = None,
        parse_mode: str = "conversation"
    ) -> Dict[str, Any]:
        """
        导入单个 session 文件

        Args:
            session_path: session 文件路径
            project_id: 项目ID
            parse_mode: 解析模式
                - "conversation": 解析对话轮次
                - "events": 解析事件消息
                - "full": 解析全部内容

        Returns:
            {
                "success": bool,
                "total_fragments": int,
                "memory_types": Dict[str, int]
            }
        """
        session_path = Path(session_path)

        if not session_path.exists():
            return {"success": False, "error": "文件不存在"}

        if parse_mode == "conversation":
            return self._parse_conversation(session_path, project_id)
        elif parse_mode == "events":
            return self._parse_events(session_path, project_id)
        else:
            return self._parse_full(session_path, project_id)

    def _parse_conversation(self, session_path: Path, project_id: str) -> Dict[str, Any]:
        """解析对话格式（user/assistant轮次）"""
        fragments = []
        types_count = {}

        with open(session_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                    payload = entry.get("payload", {})
                    content_type = payload.get("type", "")

                    # 支持多种格式
                    if content_type == "response_item":
                        role = payload.get("role")
                        content = payload.get("content", [])

                        if role == "user":
                            text = self._extract_text_from_content(content)
                            if text:
                                fragments.append({
                                    "text": text,
                                    "type": "session",
                                    "project_id": project_id
                                })
                                types_count["session"] = types_count.get("session", 0) + 1

                        elif role == "assistant":
                            # assistant 的 response 也作为 session 记录
                            for item in content:
                                if item.get("type") == "text":
                                    text = item.get("text", "")
                                    if text and len(text) > 10:
                                        fragments.append({
                                            "text": text,
                                            "type": "session",
                                            "project_id": project_id
                                        })
                                        types_count["session"] = types_count.get("session", 0) + 1

                    elif content_type in ("user_message", "message"):
                        # 支持 user/message 格式
                        text = payload.get("message", "")
                        if text and len(text) > 5:
                            fragments.append({
                                "text": text,
                                "type": "session",
                                "project_id": project_id
                            })
                            types_count["session"] = types_count.get("session", 0) + 1

                except json.JSONDecodeError:
                    # 跳过无效的 JSON
                    continue
                except Exception:
                    # 跳过其他异常
                    continue

        # 批量添加
        success = True
        for frag in fragments:
            try:
                self.memory_manager.add_fragment(
                    text=frag["text"],
                    type=frag["type"],
                    project_id=frag["project_id"]
                )
            except Exception as e:
                success = False
                print(f"添加失败: {e}")

        return {
            "success": success,
            "total_fragments": len(fragments),
            "memory_types": types_count
        }

    def _parse_events(self, session_path: Path, project_id: str) -> Dict[str, Any]:
        """解析事件消息格式"""
        fragments = []
        types_count = {}

        with open(session_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                    payload = entry.get("payload", {})
                    content_type = payload.get("type", "")

                    # 解析 event_msg
                    if content_type == "event_msg":
                        if content_type in ("user_message", "message"):
                            text = payload.get("message", "")
                            if text and len(text) > 5:
                                fragments.append({
                                    "text": text,
                                    "type": "session",
                                    "project_id": project_id
                                })
                                types_count["session"] = types_count.get("session", 0) + 1

                    # 支持 assistant_response 格式
                    elif content_type in ("assistant_response", "response"):
                        text = payload.get("content", "")
                        if text and len(text) > 5:
                            fragments.append({
                                "text": text,
                                "type": "knowledge",
                                "project_id": project_id
                            })
                            types_count["knowledge"] = types_count.get("knowledge", 0) + 1

                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue

        # 批量添加
        success = True
        for frag in fragments:
            try:
                self.memory_manager.add_fragment(
                    text=frag["text"],
                    type=frag["type"],
                    project_id=frag["project_id"]
                )
            except Exception as e:
                success = False
                print(f"添加失败: {e}")

        return {
            "success": success,
            "total_fragments": len(fragments),
            "memory_types": types_count
        }

    def _parse_full(self, session_path: Path, project_id: str) -> Dict[str, Any]:
        """解析全部内容"""
        all_text = []
        types_count = {}

        with open(session_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                entry = json.loads(line)
                payload = entry.get("payload", {})

                # 提取 text 字段
                text = payload.get("message") or payload.get("text") or ""

                if text:
                    # 自动分类
                    result = self.memory_manager.classifier.classify_auto(text)

                    if result["rule_score"] > 0:
                        mem_type = result["primary"]
                    else:
                        mem_type = "session"  # 默认

                    types_count[mem_type] = types_count.get(mem_type, 0) + 1

                    # 添加到记忆系统
                    self.memory_manager.add_fragment(
                        text=text,
                        type=mem_type,
                        project_id=project_id
                    )

        return {
            "success": True,
            "total_fragments": len(all_text),
            "memory_types": types_count
        }

    def _extract_text_from_content(self, content: List[Dict]) -> str:
        """从 content 列表中提取文本"""
        text_parts = []

        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return "\n".join(text_parts)

    def _extract_plain_text(self, text: str) -> str:
        """从复杂格式文本中提取纯文本"""
        # 移除 Markdown 格式
        text = re.sub(r'[`\*#\-\[\](){}]', '', text)
        # 移除多余空白
        text = ' '.join(text.split())
        return text.strip()


def import_sessions_batch(
    sessions_dir: str,
    memory_manager: MemoryManager,
    project_id: Optional[str] = None,
    parse_mode: str = "conversation"
) -> Dict[str, Any]:
    """
    批量导入 session 文件

    Args:
        sessions_dir: session 目录
        memory_manager: 记忆管理器
        project_id: 项目ID
        parse_mode: 解析模式

    Returns:
        导入结果汇总
    """
    sessions_dir = Path(sessions_dir)

    if not sessions_dir.exists():
        return {"success": False, "error": "目录不存在"}

    results = {
        "success": True,
        "total_files": 0,
        "total_fragments": 0,
        "memory_types": {}
    }

    # 找到所有 session 文件
    session_files = list(sessions_dir.glob("*.jsonl"))
    results["total_files"] = len(session_files)

    pipeline = SessionPipeline(memory_manager)

    for i, session_file in enumerate(session_files, 1):
        print(f"[导入 {i}/{len(session_files)}] {session_file.name}")
        result = pipeline.import_session(
            session_file,
            project_id=project_id or f"session-{i}",
            parse_mode=parse_mode
        )

        if result.get("success", False):
            results["total_fragments"] += result.get("total_fragments", 0)

            # 合并 types_count
            for mem_type, count in result.get("memory_types", {}).items():
                results["memory_types"][mem_type] = results["memory_types"].get(mem_type, 0) + count
        else:
            print(f"  错误: {result.get('error')}")

    return results
