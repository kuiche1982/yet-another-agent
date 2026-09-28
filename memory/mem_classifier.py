"""
记忆分类器 - 规则 + LLM 混合分类
使用 LFM2.5-350M-MLX-4bit 进行智能分类
支持长文本自动摘要再分类，节省 token
"""
import re
import tiktoken
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import litert_lm


class MemoryClassifier:
    """记忆分类器"""

    # 规则分类特征（按优先级排序）
    RULE_PATTERNS = [
        ("facts", [
            r"我叫", r"我叫", r"邮箱", r"偏好", r"习惯", r"不", r"不要", r"必须",
            r"我使用", r"我推荐", r"我认为", r"我喜欢", r"我的", r"个人"
        ]),
        ("config", [
            r"env\.|环境变量", r"配置文件", r"settings", r"参数", r"设置",
            r"URL=", r"PORT=", r"HOST=", r"database=", r"API_KEY="
        ]),
        ("code", [
            r"def ", r"class ", r"import ", r"from ", r"function ", r"函数", r"类",
            r"```python", r"```javascript", r"```go", r"```rust", r"```java"
        ]),
        ("knowledge", [
            r"协议规范", r"API文档", r"RFC", r"标准", r"最佳实践", r"设计模式",
            r"数据库schema", r"表结构", r"字段说明", r"规范要求"
        ]),
        ("task", [
            r"待办", r"计划", r"目标", r"下一步", r"TODO", r"FIXME", r"xxx", r"任务",
            r"完成", r"需要", r"完成", r"进行中"
        ]),
        ("session", [
            r"好的", r"收到", r"没问题", r"请问", r"谢谢", r"确认", r"同意",
            r"讨论", r"意见", r"建议", r"反馈"
        ])
    ]

    def __init__(
        self,
        use_llm: bool = True,
        model_path: str = "~/.litert-lm/models/gemma4-e4b/model.litertlm",
        # 长文本摘要配置
        summarize_threshold: int = 500,  # 超过此长度先摘要再分类
        summarize_max_tokens: int = 200,  # 摘要最大token数
        use_heavy_llm: bool = False  # 是否使用大模型（复杂场景）
    ):
        self.use_llm = use_llm
        self.model_path = model_path
        self.summarize_threshold = summarize_threshold
        self.summarize_max_tokens = summarize_max_tokens
        self.use_heavy_llm = use_heavy_llm
        self.llm = None
        self.encoder = None  # tiktoken 编码器

        if use_llm:
            self._init_llm()
            self._init_encoder()

    def _init_llm(self):
        """初始化LLM模型"""
        try:
            self.llm = litert_lm.Engine(
                self.model_path,
                backend=litert_lm.Backend.GPU(),
                vision_backend=litert_lm.Backend.CPU(),
                audio_backend=litert_lm.Backend.CPU(),
                max_num_tokens=8192,
                enable_speculative_decoding=False
            )
            model_name = "LFM2.5-350M" if "LFM" in self.model_path else "gemma4-e4b"
            print(f"[分类器] {model_name} 初始化成功")
        except Exception as e:
            print(f"[分类器] LLM初始化失败: {e}，将仅使用规则分类")
            self.use_llm = False

    def _init_encoder(self):
        """初始化 tiktoken 编码器"""
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            print("[分类器] 无法初始化 tiktoken，将使用长度估算")

    def _count_tokens(self, text: str) -> int:
        """计算文本token数"""
        if self.encoder:
            return len(self.encoder.encode(text))
        # 估算：1 token ≈ 4 characters
        return len(text) // 4

    def _summarize_text(self, text: str) -> str:
        """
        摘要长文本（仅用于分类场景）

        Args:
            text: 长文本

        Returns:
            摘要后的短文本
        """
        if not self.use_llm or self.llm is None:
            # 如果没有LLM，截断到阈值
            return text[:self.summarize_max_tokens * 4]

        try:
            prompt = f"""对下面文本生成1句话摘要，控制在 {self.summarize_max_tokens} tokens 以内：

{text[:1000]}

摘要（只返回一句话）："""

            with self.llm.create_conversation() as conv:
                resp = conv.send_message(prompt)
                summary = ""
                for c in resp.get("content", []):
                    if c.get("type") == "text":
                        summary += c["text"]
                return summary.strip()
        except Exception as e:
            print(f"[分类器] 摘要失败: {e}，截断文本")
            return text[:self.summarize_max_tokens * 4]

    def classify(self, text: str) -> List[Tuple[str, int]]:
        """
        规则分类

        Args:
            text: 要分类的文本

        Returns:
            列表：[(类型, 匹配次数), ...] 按匹配次数降序
        """
        scores = {mem_type: 0 for mem_type, _ in self.RULE_PATTERNS}

        for mem_type, patterns in self.RULE_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[mem_type] += 1

        # 过滤掉0分的类型，按分数排序
        valid_scores = [(mem_type, score) for mem_type, score in scores.items() if score > 0]
        valid_scores.sort(key=lambda x: -x[1])

        return valid_scores

    def classify_llm(self, text: str, max_length: int = 500) -> str:
        """
        LLM智能分类

        Args:
            text: 要分类的文本
            max_length: 超过此长度会截断

        Returns:
            类型名称（单数形式）
        """
        if not self.use_llm or self.llm is None:
            return "session"  # 默认回退

        # 截断文本
        text = text[:max_length]

        prompt = f"""分析下面文本属于哪种记忆类型，只返回类型名称（不要换行）：

{text}

可选类型：
1. code - 代码片段、实现细节、函数定义
2. knowledge - 规范、文档、设计说明、标准、协议
3. facts - 个人事实、属性、偏好、习惯
4. config - 配置项、参数设置、环境变量
5. session - 对话、讨论、交流、意见
6. task - 待办、计划、目标、任务

类型："""

        try:
            with self.llm.create_conversation() as conv:
                resp = conv.send_message(prompt)
                result = ""
                for c in resp.get("content", []):
                    if c.get("type") == "text":
                        result += c["text"]
            return result.strip()
        except Exception as e:
            print(f"[分类器] LLM分类失败: {e}，回退到规则分类")
            return "session"

    def _calculate_importance_score(self, text: str) -> float:
        """
        计算文本的重要性分数

        Args:
            text: 文本

        Returns:
            重要性分数 (0.0 - 1.0)
        """
        score = 0.0
        length = len(text)

        # 1. 命名实体（接口、字段、参数名）- 高权重
        if re.search(r'[A-Z][a-zA-Z0-9_]*\s*=|def\s+[a-z_]+|class\s+[A-Z]', text):
            score += 0.3

        # 2. 编号和结构化内容（1.1, 2.3, A, B, C...）- 中权重
        if re.search(r'\d+\.\d+|\w\.\w+', text):
            score += 0.2

        # 3. 关键词密度（出现关键词的词数/总词数）- 中权重
        all_keywords = [kw for _, patterns in self.RULE_PATTERNS for kw in patterns]
        text_lower = text.lower()
        keyword_count = sum(1 for kw in all_keywords if kw.lower() in text_lower)
        if length > 0:
            score += min(keyword_count * 0.05, 0.2)

        # 4. 短文本（<50字符）- 低权重
        if length < 50:
            score -= 0.1

        # 5. 长文本（>1000字符）- 高权重
        if length > 1000:
            score += 0.1

        # 6. 代码块标识（```）- 高权重
        if text.count('```') >= 2:
            score += 0.15

        return max(0.0, min(score, 1.0))

    def classify_auto(self, text: str) -> Dict[str, Any]:
        """
        自动分类（规则优先，LLM补充）

        长文本优化：超过 threshold 先摘要再分类

        Args:
            text: 要分类的文本

        Returns:
            {
                "primary": str,          # 主要类型
                "candidates": List[str],  # 候选类型
                "rule_score": int,        # 规则匹配分数
                "importance": float,      # 重要性分数
                "method": str,            # 使用的方法（rule/llm/both）
                "original_length": int,   # 原始文本长度（tokens）
                "summarized": bool        # 是否使用了摘要
            }
        """
        # 统计原始文本长度
        original_tokens = self._count_tokens(text)
        is_long_text = original_tokens > self.summarize_threshold

        # 如果是长文本，先摘要
        if is_long_text:
            print(f"[分类器] 长文本 ({original_tokens} tokens)，使用摘要优化")
            summarized_text = self._summarize_text(text)
            summarized_tokens = self._count_tokens(summarized_text)

            # 摘要结果
            result = self._classify_with_text(summarized_text, original_tokens, True)
            result["summarized"] = True
            result["summarized_tokens"] = summarized_tokens
            return result
        else:
            # 短文本直接分类
            return self._classify_with_text(text, original_tokens, False)

    def _classify_with_text(self, text: str, original_tokens: int, summarized: bool) -> Dict[str, Any]:
        """内部分类方法"""

        # 1. 规则分类
        rule_result = self.classify(text)

        if not rule_result:
            # 规则无匹配，使用LLM
            llm_type = self.classify_llm(text)
            return {
                "primary": llm_type,
                "candidates": [llm_type],
                "rule_score": 0,
                "method": "llm",
                "original_length": original_tokens,
                "summarized": summarized
            }

        # 2. 规则有匹配
        primary_type, rule_score = rule_result[0]
        candidates = [primary_type]

        # 如果有多个候选，添加LLM验证
        if len(rule_result) >= 2:
            llm_type = self.classify_llm(text)
            if llm_type != primary_type and llm_type not in candidates:
                candidates.insert(0, llm_type)  # LLM优先

        method = "rule" if len(rule_result) == 1 else "rule+llm"

        # 计算重要性分数
        importance = self._calculate_importance_score(text)

        return {
            "primary": candidates[0],
            "candidates": candidates,
            "rule_score": rule_score,
            "importance": importance,
            "method": method,
            "original_length": original_tokens,
            "summarized": summarized
        }

    def classify_batch(self, texts: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        批量分类

        Args:
            texts: 文本列表
            batch_size: 批量处理大小

        Returns:
            每个文本的分类结果列表
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                results.append(self.classify_auto(text))
        return results
