"""火影语料 multihop 测试：实体图扩展 + hybrid + rerank + 生成。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ragm.classifier.factory import build_classifier
from ragm.config import Settings
from ragm.embeddings.reranker import build_reranker
from ragm.graph.kg import KGStore
from ragm.rag.graph import RagGraph
from ragm.rag.llm import LLMClient
from ragm.store.search_service import SearchService

DATA = Path('/run/media/myj/D/rag_m/data/naruto')
QUESTIONS = [
    ('自来也的弟子中谁成为了四代火影？', ['波风水门', '水门']),
    ('晓组织里与宇智波鼬搭档的是谁？', ['干柿鬼鲛', '鬼鲛']),
    ('千手柱间的弟弟是谁？', ['千手扉间', '扉间']),
    ('谁把九尾封印在鸣人体内？', ['波风水门', '水门']),
    ('宇智波鼬的弟弟是谁？', ['佐助']),
    ('木叶三忍除了自来也和纲手还有谁？', ['大蛇丸']),
    ('谁教鸣人螺旋丸？', ['自来也']),
    ('佩恩的本体是谁？', ['长门']),
    ('四代火影的儿子是谁？', ['鸣人']),
    ('谁和宇智波斑一起建立了木叶？', ['千手柱间', '柱间']),
    ('九尾人柱力中谁成为了七代火影？', ['鸣人']),
    ('大蛇丸的弟子有哪些？', ['佐助', '兜']),
]


def main() -> None:
    base = Settings()
    settings = Settings(
        milvus_collection='rag_naruto_chunks',
        milvus_child_collection='rag_naruto_children',
        parents_path=DATA / 'parents.jsonl',
        kg_label='NarutoEntity',
        kg_relation='MENTIONS',
        rerank_enabled=True,
        rerank_model_path=Path('/run/media/myj/D/rag_m/models/rerankers/bge-reranker-base'),
    )
    classifier = build_classifier()
    search = SearchService(settings)
    kg = KGStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.kg_label, settings.kg_relation)
    kg.load()
    reranker = build_reranker('bge-reranker-base', str(settings.rerank_model_path))
    llm = LLMClient(base.llm_base_url, base.llm_model, base.llm_api_key, base.llm_temperature)
    rag = RagGraph(classifier, search, llm, kg=kg, reranker=reranker)
    rows = []
    for q, gold in QUESTIONS:
        t0 = time.perf_counter()
        state = rag.invoke(q, route_override='multihop')
        latency = round((time.perf_counter() - t0) * 1000)
        answer = (state.get('answer_lead', '') + ' ' + state.get('answer_body', '')).strip()
        ok = any(g in answer for g in gold)
        rows.append({
            'question': q, 'answer': answer[:200], 'gold': gold, 'correct': ok,
            'kg_seeds': state.get('kg_seeds') or [], 'graph_added': state.get('graph_added', 0),
            'evidence_titles': [h.get('title', '') for h in (state.get('hits') or [])][:5],
            'citations': state.get('citations') or [], 'citation_check': state.get('citation_check'),
            'latency_ms': latency,
        })
        print(json.dumps({'q': q, 'ok': ok, 'seeds': rows[-1]['kg_seeds'][:2], 'graph_added': rows[-1]['graph_added'], 'lat': latency, 'ans': answer[:60]}, ensure_ascii=False), flush=True)
    n = len(rows)
    valid = 0
    for r in rows:
        chk = r['citation_check'] or ''
        if '/' in chk:
            a, b = chk.split('/')[0], chk.split('/')[1].split()[0]
            if a == b and int(b) > 0:
                valid += 1
    summary = {'n': n, 'answer_correct': round(sum(1 for r in rows if r['correct']) / n, 3),
               'citation_all_valid': round(valid / n, 3),
               'avg_graph_added': round(sum(r['graph_added'] for r in rows) / n, 2),
               'p50_latency_ms': sorted(r['latency_ms'] for r in rows)[n // 2]}
    Path('reports/naruto_multihop.json').write_text(json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
