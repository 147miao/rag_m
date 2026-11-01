"""火影语料检索横向矩阵：embedding / 模式 / reranker / chunk size。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ragm.config import Settings
from ragm.embeddings.embedder import Embedder
from ragm.embeddings.reranker import build_reranker
from ragm.store.milvus_store import MilvusStore

DATA = Path('/run/media/myj/D/rag_m/data/naruto')
MODELS = Path('/run/media/myj/D/rag_m/models')
EVAL = [
    ('漩涡鸣人的父亲是谁？', ['波风水门'], 'fact'),
    ('九尾是什么？', ['九尾', '尾兽'], 'fact'),
    ('宇智波鼬为什么灭族？', ['宇智波鼬'], 'fact'),
    ('晓组织的目的是什么？', ['晓'], 'fact'),
    ('旗木卡卡西的写轮眼是从谁那里得到的？', ['旗木卡卡西', '宇智波带土'], 'fact'),
    ('千手柱间和宇智波斑是什么关系？', ['千手柱间', '宇智波斑'], 'fact'),
    ('尾兽一共有几只？', ['尾兽'], 'fact'),
    ('自来也有哪些弟子？', ['自来也(火影忍者)'], 'fact'),
    ('第四次忍界大战的起因是什么？', ['第四次忍界大战'], 'fact'),
    ('写轮眼和轮回眼有什么关系？', ['写轮眼', '轮回眼'], 'fact'),
    ('木叶隐村是谁建立的？', ['木叶隐村', '千手柱间'], 'fact'),
    ('六道仙人是谁？', ['六道仙人'], 'fact'),
    ('千手扉间发明了哪些忍术？', ['千手扉间'], 'fact'),
    ('九尾的人柱力有哪些？', ['人柱力', '漩涡玖辛奈'], 'fact'),
    ('晓组织的首领是谁？', ['晓', '长门(火影忍者)'], 'fact'),
    ('大筒木辉夜是什么人？', ['大筒木辉夜'], 'fact'),
    ('自来也的弟子中谁成为了四代火影？', ['自来也(火影忍者)', '波风水门'], 'multihop'),
    ('晓组织里与宇智波鼬搭档的是谁？', ['宇智波鼬', '干柿鬼鲛'], 'multihop'),
    ('千手柱间的弟弟是谁？', ['千手柱间', '千手扉间'], 'multihop'),
    ('谁把九尾封印在鸣人体内？', ['波风水门', '九尾'], 'multihop'),
    ('宇智波鼬的弟弟是谁？', ['宇智波鼬', '宇智波佐助'], 'multihop'),
    ('木叶三忍除了自来也和纲手还有谁？', ['自来也(火影忍者)', '大蛇丸(火影忍者)'], 'multihop'),
    ('谁教鸣人螺旋丸？', ['自来也(火影忍者)', '螺旋丸'], 'multihop'),
    ('佩恩的本体是谁？', ['佩恩', '长门(火影忍者)'], 'multihop'),
    ('四代火影的儿子是谁？', ['波风水门', '漩涡鸣人'], 'multihop'),
    ('谁和宇智波斑一起建立了木叶？', ['千手柱间', '宇智波斑'], 'multihop'),
    ('九尾人柱力中谁成为了七代火影？', ['漩涡鸣人', '九尾'], 'multihop'),
    ('大蛇丸的弟子有哪些？', ['大蛇丸(火影忍者)', '药师兜'], 'multihop'),
]
CHUNK_SIZES = [200, 400, 800, 1600]


def chunks_of(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def metrics(ranked, golds) -> dict:
    out = {}
    for k in (5, 10):
        recalls, ndcgs, mrrs = [], [], []
        for ranking, gold in zip(ranked, golds):
            gs = set(gold)
            top = list(dict.fromkeys(ranking))[:k]
            recalls.append(sum(1 for d in top if d in gs) / max(1, len(gs)))
            dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(top) if d in gs)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gs), k)))
            ndcgs.append(dcg / idcg if idcg else 0.0)
            mrrs.append(next((1 / (i + 1) for i, d in enumerate(top) if d in gs), 0.0))
        out[f'recall@{k}'] = round(float(np.mean(recalls)), 4)
        if k == 10:
            out['ndcg@10'] = round(float(np.mean(ndcgs)), 4)
            out['mrr@10'] = round(float(np.mean(mrrs)), 4)
    return out


def make_chunks(rows, size, overlap):
    out = []
    for r in rows:
        for i, ch in enumerate(chunks_of(r['text'], size, overlap)):
            out.append({'chunk_id': f"{r['doc_id']}::s{size}::{i}", 'doc_id': r['doc_id'], 'title': r['title'], 'source': 'moegirl', 'lang': 'zh', 'chunk_index': i, 'text': ch})
    return out


def build_index(store, embedder, chunks):
    store.recreate()
    for i in range(0, len(chunks), 256):
        batch = chunks[i:i + 256]
        vecs = embedder.encode([c["text"] for c in batch], 8)
        store.insert([{**{k: c[k] for k in ('chunk_id', 'doc_id', 'title', 'source', 'lang', 'chunk_index', 'text')}, 'dense': vecs[j].tolist()} for j, c in enumerate(batch)])
    store.flush()
    store.load()


def eval_store(store, embedder, mode='hybrid', k=10, reranker=None):
    ranked, lat = [], []
    for q, gold, _ in EVAL:
        t0 = time.perf_counter()
        qvec = embedder.encode_queries([q])[0].tolist()
        if mode == 'dense':
            hits = store.dense_search(qvec, k)
        elif mode == 'sparse':
            hits = store.sparse_search(q, k)
        else:
            hits = store.hybrid_search(q, qvec, k)
        if reranker is not None:
            hits = reranker.rerank(q, hits, top_n=k)
        lat.append((time.perf_counter() - t0) * 1000)
        ranked.append([h['title'] for h in hits])
    golds = [list(gold) for _, gold, _ in EVAL]
    m = metrics(ranked, golds)
    m['p50_ms'] = round(float(np.median(lat)), 1)
    return m


def main():
    s = Settings()
    flat = [json.loads(line) for line in (DATA / 'chunks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    results = []
    emb_models = [
        ('embeddinggemma-300m', MODELS / 'embeddings' / 'embeddinggemma-300m'),
        ('bge-m3', MODELS / 'embeddings' / 'bge-m3'),
        ('bge-large-zh-v1.5', MODELS / 'embeddings' / 'bge-large-zh-v1.5'),
        ('qwen3-embedding-0.6b', MODELS / 'embeddings' / 'qwen3-embedding-0.6b'),
    ]
    stores = {}
    gemma_emb = None
    for name, path in emb_models:
        print('== embedding', name, flush=True)
        emb = Embedder(path, name, 256)
        coll = 'nr_emb_' + name.replace('.', '_').replace('-', '_')
        store = MilvusStore(s.milvus_uri, coll, emb.dim)
        t0 = time.perf_counter()
        build_index(store, emb, flat)
        m = eval_store(store, emb, 'hybrid', 10)
        m.update({'axis': 'embedding', 'name': name, 'build_s': round(time.perf_counter() - t0, 1)})
        print(json.dumps(m, ensure_ascii=False), flush=True)
        results.append(m)
        stores[name] = store
        if name == 'embeddinggemma-300m':
            gemma_emb = emb
        else:
            del emb
        torch.cuda.empty_cache()
    store, emb = stores['embeddinggemma-300m'], gemma_emb
    for mode in ('dense', 'sparse', 'hybrid'):
        m = eval_store(store, emb, mode, 10)
        m.update({'axis': 'mode', 'name': mode})
        print(json.dumps(m, ensure_ascii=False), flush=True)
        results.append(m)
    for name, path in [('bge-reranker-base', MODELS / 'rerankers' / 'bge-reranker-base'), ('bge-reranker-v2-m3', MODELS / 'rerankers' / 'bge-reranker-v2-m3'), ('qwen3-reranker-0.6b', MODELS / 'embeddings' / 'qwen3-reranker-0.6b')]:
        print('== reranker', name, flush=True)
        rr = build_reranker(name, str(path))
        m = eval_store(store, emb, 'hybrid', 10, reranker=rr)
        m.update({'axis': 'reranker', 'name': name})
        print(json.dumps(m, ensure_ascii=False), flush=True)
        results.append(m)
        del rr
        torch.cuda.empty_cache()
    for size in CHUNK_SIZES:
        print('== chunk', size, flush=True)
        ch = make_chunks(flat, size, max(1, size // 5))
        st = MilvusStore(s.milvus_uri, f'nr_chunk_{size}', emb.dim)
        build_index(st, emb, ch)
        m = eval_store(st, emb, 'hybrid', 10)
        m.update({'axis': 'chunk_size', 'name': str(size), 'chunks': len(ch)})
        print(json.dumps(m, ensure_ascii=False), flush=True)
        results.append(m)
    Path('reports/naruto_retrieval_matrix.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved reports/naruto_retrieval_matrix.json')


if __name__ == '__main__':
    main()
