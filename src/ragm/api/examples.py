"""示例问题：按当前索引集合返回可运行的真实问题（前端左栏使用）。"""
from __future__ import annotations

EXAMPLES: dict[str, list[dict]] = {
    "rag_all_chunks": [
        {"id": "all-1", "question": "鸣人父亲是谁？", "route": "simple", "hint": "火影 · 单跳事实"},
        {"id": "all-2", "question": "自来也的弟子中谁成为了四代火影？", "route": "multihop", "hint": "火影 · 多跳"},
        {"id": "all-3", "question": "Which US documentary was made first: Scared Straight! or Soldier Girls?", "route": "multihop", "hint": "HotpotQA · 对比"},
        {"id": "all-4", "question": "What metrics does RAGAS use to evaluate RAG?", "route": "simple", "hint": "arXiv · 事实"},
        {"id": "all-5", "question": "Compare the RAG and GraphRAG approaches: when should each be used?", "route": "agentic", "hint": "arXiv · 推理"},
    ],
    "rag_naruto_chunks": [
        {"id": "nr-1", "question": "鸣人父亲是谁？", "route": "simple", "hint": "火影 · 单跳事实"},
        {"id": "nr-2", "question": "自来也的弟子中谁成为了四代火影？", "route": "multihop", "hint": "火影 · 多跳"},
        {"id": "nr-3", "question": "宇智波鼬为什么灭族？", "route": "simple", "hint": "火影 · 原因"},
        {"id": "nr-4", "question": "佩恩的本体是谁？", "route": "multihop", "hint": "火影 · 关系"},
        {"id": "nr-5", "question": "对比宇智波斑和千手柱间的关系与目标差异", "route": "agentic", "hint": "火影 · 推理"},
    ],
    "rag_papers_chunks": [
        {"id": "pp-1", "question": "What metrics does RAGAS use to evaluate RAG?", "route": "simple", "hint": "arXiv · 事实"},
        {"id": "pp-2", "question": "What problem does RAG address for language models?", "route": "simple", "hint": "arXiv · 事实"},
        {"id": "pp-3", "question": "How does Self-RAG decide when to retrieve?", "route": "multihop", "hint": "arXiv · 机制"},
        {"id": "pp-4", "question": "What is the main advantage of GraphRAG over baseline RAG?", "route": "multihop", "hint": "arXiv · 对比"},
        {"id": "pp-5", "question": "Compare Corrective RAG and Self-RAG: how do they handle poor retrieval?", "route": "agentic", "hint": "arXiv · 推理"},
    ],
    "rag_chunks": [
        {"id": "dm-1", "question": "Which magazine was started first, Arthur's Magazine or First for Women?", "route": "multihop", "hint": "HotpotQA · 对比"},
        {"id": "dm-2", "question": "范廷颂是什么时候被任为主教的？", "route": "simple", "hint": "CMRC · 事实"},
        {"id": "dm-3", "question": "Who wrote the novel that the film Svengali is based on?", "route": "multihop", "hint": "HotpotQA · 多跳"},
        {"id": "dm-4", "question": "Which US documentary was made first: Scared Straight! or Soldier Girls?", "route": "multihop", "hint": "HotpotQA · 对比"},
    ],
}

DEFAULT_EXAMPLES = EXAMPLES["rag_all_chunks"]


def examples_for(collection: str) -> list[dict]:
    return EXAMPLES.get(collection, DEFAULT_EXAMPLES)