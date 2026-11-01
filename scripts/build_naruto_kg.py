"""构建火影语料实体图：节点=条目，边=条目正文中提及的其他条目。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

RAW = Path('/run/media/myj/D/rag_m/data/naruto/moegirl.jsonl')
OUT = Path('/run/media/myj/D/rag_m/data/naruto')

TYPE_RULES = [
    ('忍术', ['之术', '螺旋丸', '千鸟', '雷切', '影分身', '须佐能乎', '通灵', '幻术', '体术', '忍术']),
    ('组织', ['晓', '木叶', '忍村', '联军', '七人众', '暗部', '根', '组织', '一族']),
    ('地点', ['村', '山', '洞', '林', '国', '之里', '隐村']),
    ('设定', ['尾兽', '查克拉', '写轮眼', '轮回眼', '白眼', '人柱力', '血继', '咒印', '神树', '十尾']),
]


def entity_type(title: str) -> str:
    for t, keys in TYPE_RULES:
        if any(k in title for k in keys):
            return t
    return '人物'


def main() -> None:
    rows = [json.loads(line) for line in RAW.read_text(encoding='utf-8').splitlines() if line.strip()]
    titles = {r['title']: r for r in rows}
    names = [t for t in titles if 2 <= len(t) <= 30]
    nodes = []
    for t in names:
        nodes.append({'name': t, 'type': entity_type(t), 'doc_id': 'nr_' + hashlib.sha1(t.encode()).hexdigest()[:12], 'chars': len(titles[t]['text'])})
    edges = []
    for t in names:
        text = titles[t]['text']
        for other in names:
            if other == t:
                continue
            if other in text:
                edges.append({'a': t, 'b': other, 'qtype': 'mention', 'question': f'{t} 条目提到了 {other}'})
    for name, data in (('kg_nodes', nodes), ('kg_edges', edges)):
        (OUT / f'{name}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in data), encoding='utf-8')
    print(json.dumps({'nodes': len(nodes), 'edges': len(edges)}, ensure_ascii=False))


if __name__ == '__main__':
    main()