"""火影语料生成质量矩阵（通过 API，rerank 由环境变量控制）。"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ragm.config import get_settings
from ragm.rag.llm import LLMClient

API = 'http://127.0.0.1:8000/api/query'
EVAL = [
    ('漩涡鸣人的父亲是谁？', ['波风水门'], 'fact'),
    ('九尾是什么？', ['九尾', '尾兽'], 'fact'),
    ('宇智波鼬为什么灭族？', ['宇智波鼬'], 'fact'),
    ('晓组织的目的是什么？', ['晓'], 'fact'),
    ('旗木卡卡西的写轮眼是从谁那里得到的？', ['卡卡西', '带土'], 'fact'),
    ('千手柱间和宇智波斑是什么关系？', ['柱间', '斑'], 'fact'),
    ('尾兽一共有几只？', ['九只', '9只'], 'fact'),
    ('自来也有哪些弟子？', ['鸣人', '水门', '长门'], 'fact'),
    ('第四次忍界大战的起因是什么？', ['带土', '斑', '月之眼'], 'fact'),
    ('写轮眼和轮回眼有什么关系？', ['进化', '轮回眼'], 'fact'),
    ('木叶隐村是谁建立的？', ['柱间', '斑'], 'fact'),
    ('六道仙人是谁？', ['羽衣', '大筒木'], 'fact'),
    ('千手扉间发明了哪些忍术？', ['影分身', '飞雷神', '秽土转生'], 'fact'),
    ('九尾的人柱力有哪些？', ['水门', '玖辛奈', '鸣人'], 'fact'),
    ('晓组织的首领是谁？', ['长门', '佩恩'], 'fact'),
    ('大筒木辉夜是什么人？', ['辉夜', '查克拉'], 'fact'),
    ('自来也的弟子中谁成为了四代火影？', ['波风水门', '水门'], 'multihop'),
    ('晓组织里与宇智波鼬搭档的是谁？', ['干柿鬼鲛', '鬼鲛'], 'multihop'),
    ('千手柱间的弟弟是谁？', ['扉间'], 'multihop'),
    ('谁把九尾封印在鸣人体内？', ['波风水门', '水门'], 'multihop'),
    ('宇智波鼬的弟弟是谁？', ['佐助'], 'multihop'),
    ('木叶三忍除了自来也和纲手还有谁？', ['大蛇丸'], 'multihop'),
    ('谁教鸣人螺旋丸？', ['自来也'], 'multihop'),
    ('佩恩的本体是谁？', ['长门'], 'multihop'),
    ('四代火影的儿子是谁？', ['鸣人'], 'multihop'),
    ('谁和宇智波斑一起建立了木叶？', ['柱间'], 'multihop'),
    ('九尾人柱力中谁成为了七代火影？', ['鸣人'], 'multihop'),
    ('大蛇丸的弟子有哪些？', ['佐助', '兜'], 'multihop'),
]

_llm = None


def llm():
    global _llm
    if _llm is None:
        s = get_settings()
        _llm = LLMClient(s.llm_base_url, s.llm_model, s.llm_api_key, 0.0)
    return _llm


def ask_json(prompt):
    for _ in range(2):
        try:
            out = llm().chat([{'role': 'user', 'content': prompt}], max_tokens=800)
            raw = out['text'].strip()
            fence = chr(96) * 3
            raw = re.sub(r'^' + fence + r'[a-zA-Z]*\n?', '', raw)
            raw = re.sub(r'\n?' + fence + r'$', '', raw).strip()
            m = re.search(r'\{.*\}', raw, re.S)
            if m:
                return json.loads(m.group(0))
        except Exception:
            time.sleep(1)
    return {}


def query_api(q):
    body = json.dumps({'question': q}).encode()
    last = None
    for _ in range(3):
        try:
            req = urllib.request.Request(API, data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=240) as resp:
                return json.load(resp)
        except Exception as exc:
            last = exc
            time.sleep(2)
    raise last


def normalize(t):
    return re.sub(r'\s+', ' ', (t or '').lower()).strip()


def eval_one(item):
    q, gold, typ = item
    t0 = time.perf_counter()
    resp = query_api(q)
    latency = (time.perf_counter() - t0) * 1000
    answer = (resp['answer']['lead'] + ' ' + resp['answer']['body']).strip()
    contexts = [(e.get('full_text') or e.get('snippet') or '') for e in resp['evidence'][:8]]
    ok = any(g in answer for g in gold)
    faith = prec = 0.0
    if answer:
        ctx = '\n'.join(f'[{i}] {c[:600]}' for i, c in enumerate(contexts, start=1))
        out = ask_json(f'你是严格的 RAG 评测员。\n【问题】{q}\n【答案】{answer}\n【检索上下文】\n{ctx}\n\n只输出 JSON：{{"claims":[{{"text":"...","supported":true}}],"context_relevant":[1,0]}}')
        claims = out.get('claims') or []
        rel = out.get('context_relevant') or []
        faith = sum(1 for c in claims if c.get('supported')) / len(claims) if claims else 0.0
        prec = sum(rel) / len(rel) if rel else 0.0
    recall = 1.0 if any(normalize(g) in normalize(' '.join(contexts)) for g in gold) else 0.0
    check = resp['stats']['citationCheck']
    num, den = (check.split('通过')[0].strip().split('/') + ['0', '0'])[:2]
    return {'question': q, 'type': typ, 'correct': ok, 'faithfulness': round(faith, 4), 'context_precision': round(prec, 4),
            'context_recall': recall, 'citation_valid': 1.0 if int(den) > 0 and num == den else 0.0,
            'latency_ms': round(latency), 'route': resp['route']}


def main():
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else 'default'
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(eval_one, EVAL))
    keys = ['correct', 'faithfulness', 'context_precision', 'context_recall', 'citation_valid']
    summary = {'n': len(rows), 'config': tag, 'elapsed_s': round(time.perf_counter() - t0, 1)}
    for k in keys:
        summary[k] = round(sum(r[k] for r in rows) / len(rows), 4)
    for typ in ('fact', 'multihop'):
        sub = [r for r in rows if r['type'] == typ]
        summary[typ + '_correct'] = round(sum(1 for r in sub if r['correct']) / len(sub), 4)
    summary['p50_latency_ms'] = sorted(r['latency_ms'] for r in rows)[len(rows) // 2]
    Path(f'reports/naruto_generation_{tag}.json').write_text(json.dumps({'summary': summary, 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
