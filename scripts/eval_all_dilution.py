"""检索稀释评估：火影 28 题在「火影专用索引」 vs 「全量索引」下的检索指标。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ragm.config import Settings
from ragm.embeddings.embedder import Embedder
from ragm.store.milvus_store import MilvusStore

DATA = Path("/run/media/myj/D/rag_m/data")
MODELS = Path("/run/media/myj/D/rag_m/models")
EVAL = [
    ("漩涡鸣人的父亲是谁？", ["波风水门"]),
    ("九尾是什么？", ["九尾", "尾兽"]),
    ("宇智波鼬为什么灭族？", ["宇智波鼬"]),
    ("晓组织的目的是什么？", ["晓"]),
    ("旗木卡卡西的写轮眼是从谁那里得到的？", ["旗木卡卡西", "宇智波带土"]),
    ("千手柱间和宇智波斑是什么关系？", ["千手柱间", "宇智波斑"]),
    ("尾兽一共有几只？", ["尾兽"]),
    ("自来也有哪些弟子？", ["自来也(火影忍者)"]),
    ("第四次忍界大战的起因是什么？", ["第四次忍界大战"]),
    ("写轮眼和轮回眼有什么关系？", ["写轮眼", "轮回眼"]),
    ("木叶隐村是谁建立的？", ["木叶隐村", "千手柱间"]),
    ("六道仙人是谁？", ["六道仙人"]),
    ("千手扉间发明了哪些忍术？", ["千手扉间"]),
    ("九尾的人柱力有哪些？", ["人柱力", "漩涡玖辛奈"]),
    ("晓组织的首领是谁？", ["晓", "长门(火影忍者)"]),
    ("大筒木辉夜是什么人？", ["大筒木辉夜"]),
    ("自来也的弟子中谁成为了四代火影？", ["自来也(火影忍者)", "波风水门"]),
    ("晓组织里与宇智波鼬搭档的是谁？", ["宇智波鼬", "干柿鬼鲛"]),
    ("千手柱间的弟弟是谁？", ["千手柱间", "千手扉间"]),
    ("谁把九尾封印在鸣人体内？", ["波风水门", "九尾"]),
    ("宇智波鼬的弟弟是谁？", ["宇智波鼬", "宇智波佐助"]),
    ("木叶三忍除了自来也和纲手还有谁？", ["自来也(火影忍者)", "大蛇丸(火影忍者)"]),
    ("谁教鸣人螺旋丸？", ["自来也(火影忍者)", "螺旋丸"]),
    ("佩恩的本体是谁？", ["佩恩", "长门(火影忍者)"]),
    ("四代火影的儿子是谁？", ["波风水门", "漩涡鸣人"]),
    ("谁和宇智波斑一起建立了木叶？", ["千手柱间", "宇智波斑"]),
    ("九尾人柱力中谁成为了七代火影？", ["漩涡鸣人", "九尾"]),
    ("大蛇丸的弟子有哪些？", ["大蛇丸(火影忍者)", "药师兜"]),
]


def metrics(ranked, golds):
    out = {}
    for k in (5, 10):
        rec, nd, mrr = [], [], []
        for r, g in zip(ranked, golds):
            gs = set(g)
            top = list(dict.fromkeys(r))[:k]
            rec.append(sum(1 for d in top if d in gs) / max(1, len(gs)))
            dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(top) if d in gs)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gs), k)))
            nd.append(dcg / idcg if idcg else 0.0)
            mrr.append(next((1 / (i + 1) for i, d in enumerate(top) if d in gs), 0.0))
        out["recall@%d" % k] = round(float(np.mean(rec)), 4)
        if k == 10:
            out["ndcg@10"] = round(float(np.mean(nd)), 4)
            out["mrr@10"] = round(float(np.mean(mrr)), 4)
    return out


def main() -> None:
    s = Settings()
    emb = Embedder(MODELS / "embeddings" / "embeddinggemma-300m", "embeddinggemma-300m", 512)
    golds = [g for _, g in EVAL]
    for coll in ("rag_naruto_chunks", "rag_all_chunks"):
        st = MilvusStore(s.milvus_uri, coll, emb.dim)
        ranked = []
        for q, _g in EVAL:
            qv = emb.encode_queries([q])[0].tolist()
            hits = st.hybrid_search(q, qv, 10)
            ranked.append([h["title"] for h in hits])
        m = metrics(ranked, golds)
        m["collection"] = coll
        print(json.dumps(m, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()