"""构建火影忍者语料：parents/children/chunks（small-to-big + flat hybrid）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RAW = Path('/run/media/myj/D/rag_m/data/naruto/moegirl.jsonl')
OUT = Path('/run/media/myj/D/rag_m/data/naruto')
PARENT_MAX = 1600
CHILD = 200
CHILD_OVERLAP = 40
FLAT = 800
FLAT_OVERLAP = 100


def chunks_of(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return out


def main() -> None:
    rows = [json.loads(line) for line in RAW.read_text(encoding='utf-8').splitlines() if line.strip()]
    children, parents, chunks = [], [], []
    for r in rows:
        title = r['title']
        doc_id = 'nr_' + hashlib.sha1(title.encode()).hexdigest()[:12]
        text = r['text'].strip()
        if len(text) < 300:
            continue
        for pi, ptext in enumerate(chunks_of(text, PARENT_MAX, 0)):
            parent_id = f'{doc_id}::p{pi}'
            parents.append({'parent_id': parent_id, 'doc_id': doc_id, 'title': title, 'source': 'moegirl', 'lang': 'zh', 'text': ptext})
            for ci, ctext in enumerate(chunks_of(ptext, CHILD, CHILD_OVERLAP)):
                children.append({'chunk_id': f'{parent_id}::c{ci}', 'parent_id': parent_id, 'doc_id': doc_id, 'title': title, 'source': 'moegirl', 'lang': 'zh', 'chunk_index': ci, 'text': ctext})
        for fi, ftext in enumerate(chunks_of(text, FLAT, FLAT_OVERLAP)):
            chunks.append({'chunk_id': f'{doc_id}::f{fi}', 'doc_id': doc_id, 'title': title, 'source': 'moegirl', 'lang': 'zh', 'chunk_index': fi, 'text': ftext})
    for name, data in (('parents', parents), ('children', children), ('chunks', chunks)):
        (OUT / f'{name}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in data), encoding='utf-8')
    stats = {'docs': len(rows), 'parents': len(parents), 'children': len(children), 'chunks': len(chunks), 'avg_parent': round(sum(len(p['text']) for p in parents) / max(1, len(parents)), 1)}
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    main()