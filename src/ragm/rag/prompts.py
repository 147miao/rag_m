"""生成阶段提示词：编号上下文 + 强制引用 + 拒答策略。

输出格式（纯文本，避免 JSON 转义问题）：
  第一行：一句话结论（<=60 字，可带 [n]）
  空行
  后续：2-5 句展开（带 [n]）
"""
from __future__ import annotations

SYSTEM_PROMPT = """你是 RAG-M 的检索增强问答助手。必须遵守：
1. 只依据【检索资料】回答，不得使用资料之外的知识，不得编造事实、数字或引用编号。
2. 每个关键结论后面用 [n] 标注来源编号（n 是资料序号），可以连续标注多个，例如 [1][3]。
3. 资料不足以回答时，明确说"资料中没有足够信息"，不要猜测。
4. 用与【问题】相同的语言回答（问题为英文则整段用英文，中文则整段用中文，不要中英混用），格式固定：
   第一行：一句话结论（不超过 60 字，可带 [n]）
   空行
   之后：2-5 句展开，每句关键结论后带 [n]
5. 不要输出 JSON、Markdown 标题、代码块或"结论："前缀。"""

ROUTE_HINT = {
    "simple": "这是单跳事实问题，直接给出明确答案即可。",
    "multihop": "这是多跳问题，需要串联多条资料（先定位 A，再由 A 推出 B），请在展开部分写清推理链条并逐条引用。",
    "agentic": "这是复杂问题，需要综合多条资料对比/归纳后给出结论，明确区分事实与推断。",
}


def _context_lines(hits: list[dict]) -> list[str]:
    lines = ["【检索资料】"]
    for i, h in enumerate(hits, start=1):
        text = " ".join(str(h.get("text", "")).split())[:1200]
        lines.append(f"[{i}] 标题：{h.get('title', '')} | 来源：{h.get('source', '')} | 语言：{h.get('lang', '')}")
        lines.append(f"    内容：{text}")
    return lines


def _question_language(question: str) -> str:
    return "中文" if any("\u4e00" <= ch <= "\u9fff" for ch in question) else "英文"


def build_messages(
    question: str,
    hits: list[dict],
    route: str,
    kg_paths: list[dict] | None = None,
    web_results: list[dict] | None = None,
) -> list[dict]:
    lines: list[str] = []
    if kg_paths:
        lines.append("【图谱关系】（实体间的多跳连接，来自知识图谱）")
        for i, e in enumerate(kg_paths[:12], start=1):
            q = " ".join(str(e.get("question") or "").split())[:90]
            lines.append(f"  G{i}. {e.get('a')} —[{e.get('qtype') or 'co_evidence'}]→ {e.get('b')}（依据问题：{q}）")
        lines.append("")
    lines.extend(_context_lines(hits))
    if web_results:
        lines.append("")
        lines.append("【联网结果】（语料库之外的网页摘要，引用时用资料编号）")
        for i, w in enumerate(web_results[:6], start=1):
            lines.append(f"  W{i}. {w.get('title', '')} | {w.get('url', '')}")
            lines.append(f"      {str(w.get('snippet', ''))[:200]}")
    lines.append("")
    lines.append(f"【问题】{question}")
    lines.append(f"【路由提示】{ROUTE_HINT.get(route, ROUTE_HINT['simple'])}")
    lines.append(f"【输出语言】必须用{_question_language(question)}回答，整段答案不得混用其他语言。")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def build_messages_plain(question: str, hits: list[dict], route: str, kg_paths: list[dict] | None = None) -> list[dict]:
    return build_messages(question, hits, route, kg_paths)