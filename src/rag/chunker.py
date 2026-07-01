"""文档切割器 —— 入库前把长文本切成可检索的片段。

支持两种策略：
1. sliding_window: 固定窗口 + 重叠滑动（当前使用）
2. recursive: 按分隔符优先级递归切分（更智能，需要时切换）

- 为什么要有 overlap：防止关键信息恰好落在切割边界被截断
- 为什么按分隔符优先级切：保证片段是完整语义单元，不会把"烤鸡"切成"烤"+"鸡"
- chunk_size 怎么定：embedding 模型的 max_tokens 上限减去安全余量
"""

import re
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """切割配置"""
    chunk_size: int = 128       # 目标片段大小（字符数）
    chunk_overlap: int = 32     # 相邻片段重叠量
    strategy: str = "sliding_window"  # sliding_window | recursive


class TextChunker:
    """文本切割器"""

    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()

    # ==================== 对外接口 ====================
    def chunk(self, text: str) -> list[str]:
        """入口：根据config.strategy选择切割方式"""
        if self.config.strategy == "recursive":
            return self._recursive_chunk(text)
        return self._sliding_window_chunk(text)

    # ============策略1：滑动窗口切割================
    def _sliding_window_chunk(self, text: str) -> list[str]:
        """固定大小窗口 + overlap 滑动。

        示例：text="ABCDEFGH"，chunk_size=4, overlap=2
        → ["ABCD", "CDEF", "EFGH"]
        """
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        if len(text) <= size:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap  # 向前滑动时保留overlap区域
        return chunks

    # ============策略2：递归切割================
    # 分隔符优先级：段落 → 句子 → 短语 → 词 → 字符
    _SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]

    def _recursive_chunk(self, text: str) -> list[str]:
        """按分隔符优先级逐级尝试切割。

        思路：
        1. 先用 \n\n（段落）切 → 每段 <= chunk_size 就保留
        2. 超出的段落再用 \n（行）切
        3. 再超出的用 。！？（句子）切
        4. ...最终兜底用字符切

        这样保证片段永远断在自然标点处，不会被"烤"和"鸡"分开。
        """
        return self._split_recursive(text, self._SEPARATORS)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        seps = separators.copy()
        sep = seps.pop(0) if seps else ""
        # 用当前分隔符切分
        if sep:
            parts = re.split(f"({re.escape(sep)})", text)
        else:
            parts = [text]
        chunks: list[str] = []
        current = ""
        for part in parts:
            # 切割符自身粘在前一段末尾
            candidate = current + part
            # 纯切割符的片段直接拼入，不另起一段
            if sep and part == sep:
                current = candidate
                continue
            if len(candidate) <= self.config.chunk_size:
                current = candidate
            else:
                # 当前buffer先存
                if current.strip():
                    chunks.append(current)
                # 超长部分递归下一级分隔符切
                if seps:
                    chunks.extend(self._split_recursive(part, seps))
                else:
                    # 兜底：按字符切
                    chunks.extend(self._sliding_window_chunk(part))
                current = ""
        if current.strip():
            chunks.append(current)
        return chunks

    # ==================== Token 级切割（可选） ====================

    def chunk_by_tokens(self, text: str, tokenizer, max_tokens: int = 512) -> list[str]:
        """按 Token 数切割 —— 需要传 tokenizer（如 tiktoken）。

        Token 级切割比字符级精确：embedding 模型按 token 计费/截断，
        用 token 数控制片段大小不会超出模型限制。
        """
        tokens = tokenizer.encode(text)
        overlap = self.config.chunk_overlap

        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunks.append(tokenizer.decode(tokens[start:end]))
            if end >= len(tokens):
                break
            start = end - overlap
        return chunks


# ==================== 自测 ====================
if __name__ == "__main__":
    chunker = TextChunker(ChunkConfig(chunk_size=30, chunk_overlap=8))

    # 测试 1: 滑动窗口
    text1 = "烤鸡是一道经典中式菜肴，以整鸡为主料，经过腌制和烤制而成。外皮酥脆，肉质鲜嫩多汁。"
    print("=== 滑动窗口 (size=30, overlap=8) ===")
    for i, c in enumerate(chunker._sliding_window_chunk(text1)):
        print(f"  [{i}] {c}")

    # 测试 2: 递归切割
    print("\n=== 递归切割 (size=30) ===")
    chunker.config.strategy = "recursive"
    for i, c in enumerate(chunker.chunk(text1)):
        print(f"  [{i}] {c}")

    # 测试 3: 短文本（不满一个 chunk）
    print(f"\n=== 短文本 ===")
    print(f"  chunk('烤鸡') = {chunker.chunk('烤鸡')}")
