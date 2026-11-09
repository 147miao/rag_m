"""火影语料：答案 span 级 gold 重跑 reranker / chunk-size 对照。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ragm.config import Settings
from ragm.embeddings.embedder import Embedder
from ragm.embeddings.reranker import build_reranker
from ragm.store.milvus_store import MilvusStore

DATA = Path("/run/media/myj/D/rag_m/data/naruto")
MODELS = Path("/run/media/myj/D/rag_m/models")
Q = [
    ("漩涡鸣人的父亲是谁？", ["波风水门"], ["波风水门"]),
    ("九尾是什么？", ["九尾", "尾兽"], ["九尾", "尾兽"]),
    ("宇智波鼬为什么灭族？", ["宇智波鼬"], ["灭族", "村子", "政变"]),
    ("晓组织的目的是什么？", ["晓"], ["尾兽", "月之眼", "收集"]),
    ("旗木卡卡西的写轮眼是从谁那里得到的？", ["旗木卡卡西", "宇智波带土"], ["带土", "写轮眼"]),
    ("千手柱间和宇智波斑是什么关系？", ["千手柱间", "宇智波斑"], ["创立", "木叶", "对手"]),
    ("尾兽一共有几只？", ["尾兽"], ["九只", "九大"]),
    ("自来也有哪些弟子？", ["自来也(火影忍者)"], ["弟子", "水门", "鸣人", "长门"]),
    ("第四次忍界大战的起因是什么？", ["第四次忍界大战"], ["带土", "月之眼", "尾兽"]),
    ("写轮眼和轮回眼有什么关系？", ["写轮眼", "轮回眼"], ["进化", "轮回眼"]),
    ("木叶隐村是谁建立的？", ["木叶隐村", "千手柱间"], ["柱间", "斑", "建立"]),
    ("六道仙人是谁？", ["六道仙人"], ["羽衣", "大筒木"]),
    ("千手扉间发明了哪些忍术？", ["千手扉间"], ["影分身", "飞雷神", "秽土转生"]),
    ("九尾的人柱力有哪些？", ["人柱力", "漩涡玖辛奈"], ["人柱力", "玖辛奈", "鸣人"]),
    ("晓组织的首领是谁？", ["晓", "长门(火影忍者)"], ["长门", "佩恩", "弥彦"]),
    ("大筒木辉夜是什么人？", ["大筒木辉夜"], ["查克拉", "辉夜", "神树"]),
    ("自来也的弟子中谁成为了四代火影？", ["自来也(火影忍者)", "波风水门"], ["水门", "四代"]),
    ("晓组织里与宇智波鼬搭档的是谁？", ["宇智波鼬", "干柿鬼鲛"], ["鬼鲛", "搭档"]),
    ("千手柱间的弟弟是谁？", ["千手柱间", "千手扉间"], ["扉间", "弟弟"]),
    ("谁把九尾封印在鸣人体内？", ["波风水门", "九尾"], ["水门", "封印", "九尾"]),
    ("宇智波鼬的弟弟是谁？", ["宇智波鼬", "宇智波佐助"], ["佐助", "弟弟"]),
    ("木叶三忍除了自来也和纲手还有谁？", ["自来也(火影忍者)", "大蛇丸(火影忍者)"], ["大蛇丸", "三忍"]),
    ("谁教鸣人螺旋丸？", ["自来也(火影忍者)", "螺旋丸"], ["自来也", "螺旋丸"]),
    ("佩恩的本体是谁？", ["佩恩", "长门(火影忍者)"], ["长门", "本体"]),
    ("四代火影的儿子是谁？", ["波风水门", "漩涡鸣人"], ["鸣人", "儿子"]),
    ("谁和宇智波斑一起建立了木叶？", ["千手柱间", "宇智波斑"], ["柱间", "建立", "木叶"]),
    ("九尾人柱力中谁成为了七代火影？", ["漩涡鸣人", "九尾"], ["鸣人", "七代"]),
    ("大蛇丸的弟子有哪些？", ["大蛇丸(火影忍者)", "药师兜"], ["佐助", "兜", "弟子"]),
]


def chunks_of(text: str, size: int, overlap: int):
    text = text.strip()
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def metrics(ranked, golds):
    out = {}
    for k in (5, 10):
        recalls, ndcgs, mrrs = [], [], []
        for ranking, gold in zip(ranked, golds):
            gs = set(gold)
            top = ranking[:k]
            recalls.append(sum(1 for d in top if d in gs) / max(1, len(gs)))
            dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(top) if d in gs)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gs), k)))
            ndcgs.append(dcg / idcg if idcg else 0.0)
            mrrs.append(next((1 / (i + 1) for i, d in enumerate(top) if d in gs), 0.0))
        out[f"recall@{k}"] = round(float(np.mean(recalls)), 4)
        if k == 10:
            out["ndcg@10"] = round(float(np.mean(ndcgs)), 4)
            out["mrr@10"] = round(float(np.mean(mrrs)), 4)
    return out


def gold_ids_for(size: int, overlap: int, docs: dict):
    golds = []
    for q, titles, keys in Q:
        ids = []
        for t in titles:
            d = docs.get(t)
            if not d:
                continue
            for i, ch in enumerate(chunks_of(d["text"], size, overlap)):
                if any(k in ch for k in keys):
                    doc_id = "nr_" + hashlib.sha1(t.encode()).hexdigest()[:12]
                    ids.append(doc_id + "::s" + str(size) + "::" + str(i))
        golds.append(ids)
    return golds


def main() -> None:
    s = Settings()
    docs = {r["title"]: r for r in (json.loads(l) for l in (DATA / "moegirl.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    results = []
    emb = Embedder(MODELS / "embeddings" / "embeddinggemma-300m", "embeddinggemma-300m", 512)
    store = MilvusStore(s.milvus_uri, "rag_naruto_chunks", emb.dim)
    golds = gold_ids_for(800, 160, docs)
    for name, path in [("none", None), ("bge-reranker-base", MODELS / "rerankers" / "bge-reranker-base"), ("bge-reranker-v2-m3", MODELS / "rerankers" / "bge-reranker-v2-m3"), ("qwen3-reranker-0.6b", MODELS / "embeddings" / "qwen3-reranker-0.6b")]:
        rr = None if path is None else build_reranker(name, str(path))
        ranked = []
        for q, titles, keys in Q:
            qv = emb.encode_queries([q])[0].tolist()
            hits = store.hybrid_search(q, qv, 20)
            if rr is not None:
                hits = rr.rerank(q, hits, top_n=20)
            ranked.append([h["chunk_id"] for h in hits])
        m = metrics(ranked, golds)
        m.update({"axis": "reranker", "name": name})
        print(json.dumps(m, ensure_ascii=False), flush=True)
        results.append(m)
        if rr is not None:
            del rr
            import torch
            torch.cuda.empty_cache()
    for size in (200, 400, 800, 1600):
        st = MilvusStore(s.milvus_uri, "nr_chunk_" + str(size), emb.dim)
        g = gold_ids_for(size, max(1, size // 5), docs)
        ranked = []
        for q, titles, keys in Q:
            qv = emb.encode_queries([q])[0].tolist()
            hits = st.hybrid_search(q, qv, 20)
            ranked.append([h["chunk_id"] for h in hits])
        m = metrics(ranked, g)
        m.update({"axis": "chunk_size", "name": str(size)})
        print(json.dumps(m, ensure_ascii=False), flush=True)
        results.append(m)
    Path("reports/naruto_span_matrix.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    main()