"""用真实抓取的火影忍者资料测试 RAG 全链路。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ragm.classifier.factory import build_classifier
from ragm.config import Settings
from ragm.embeddings.reranker import build_reranker
from ragm.rag.graph import RagGraph
from ragm.rag.llm import LLMClient
from ragm.store.search_service import SearchService

DATA = Path('/run/media/myj/D/rag_m/data/naruto')
QUESTIONS = [
    ('漩涡鸣人的父亲是谁？', ['波风水门']),
    ('九尾是什么？', ['尾兽', '九喇嘛']),
    ('宇智波鼬为什么灭族？', ['保护', '村子', '木叶']),
    ('晓组织的目的是什么？', ['尾兽', '月之眼', '收集']),
    ('旗木卡卡西的写轮眼是从谁那里得到的？', ['带土']),
    ('千手柱间和宇智波斑是什么关系？', ['木叶', '创立', '对手', '朋友']),
    ('火影是什么职位？', ['首领', '村长', '木叶']),
    ('尾兽一共有几只？', ['九只', '9只', '九大']),
    ('自来也有哪些弟子？', ['鸣人', '水门', '长门']),
    ('第四次忍界大战的起因是什么？', ['带土', '斑', '月之眼', '尾兽']),
    ('写轮眼和轮回眼有什么关系？', ['进化', '六道', '轮回眼']),
    ('木叶隐村是谁建立的？', ['柱间', '斑']),
    ('六道仙人是谁？', ['羽衣', '大筒木']),
    ('千手扉间发明了哪些忍术？', ['影分身', '飞雷神', '秽土转生']),
    ('九尾的人柱力有哪些？', ['水门', '玖辛奈', '鸣人']),
    ('晓组织的首领是谁？', ['长门', '佩恩']),
    ('忍者联军有多少人？', ['八万', '8万', '八万人']),
    ('大筒木辉夜是什么人？', ['辉夜', '查克拉', '六道']),
]


def main() -> None:
    base = Settings()
    settings = Settings(
        milvus_collection='rag_naruto_chunks',
        milvus_child_collection='rag_naruto_children',
        parents_path=DATA / 'parents.jsonl',
        rerank_enabled=True,
        rerank_model_path=Path('/run/media/myj/D/rag_m/models/rerankers/bge-reranker-base'),
    )
    classifier = build_classifier()
    search = SearchService(settings)
    reranker = build_reranker('bge-reranker-base', str(settings.rerank_model_path))
    llm = LLMClient(base.llm_base_url, base.llm_model, base.llm_api_key, base.llm_temperature)
    rag = RagGraph(classifier, search, llm, kg=None, reranker=reranker)
    rows = []
    for q, gold in QUESTIONS:
        t0 = time.perf_counter()
        state = rag.invoke(q)
        latency = round((time.perf_counter() - t0) * 1000)
        answer = (state.get('answer_lead', '') + ' ' + state.get('answer_body', '')).strip()
        ok = any(g in answer for g in gold)
        titles = [h.get('title', '') for h in (state.get('hits') or [])][:5]
        rows.append({'question': q, 'answer': answer[:200], 'gold': gold, 'correct': ok,
                     'route': state.get('route'), 'citations': state.get('citations') or [],
                     'citation_check': state.get('citation_check'), 'evidence_titles': titles, 'latency_ms': latency})
        print(json.dumps({'q': q, 'ok': ok, 'route': state.get('route'), 'lat': latency, 'ans': answer[:70]}, ensure_ascii=False), flush=True)
    n = len(rows)
    valid = 0
    for r in rows:
        chk = r['citation_check'] or ''
        if '/' in chk:
            a, b = chk.split('/')[0], chk.split('/')[1].split()[0]
            if a == b and int(b) > 0:
                valid += 1
    summary = {'n': n, 'answer_correct': round(sum(1 for r in rows if r['correct']) / n, 3),
               'citation_all_valid': round(valid / n, 3), 'p50_latency_ms': sorted(r['latency_ms'] for r in rows)[n // 2]}
    Path('reports/naruto_test.json').write_text(json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
