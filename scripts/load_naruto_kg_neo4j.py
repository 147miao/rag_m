"""把火影实体图导入 Neo4j（label=NarutoEntity, relation=MENTIONS）。"""
from __future__ import annotations

import json
from pathlib import Path

from neo4j import GraphDatabase

OUT = Path('/run/media/myj/D/rag_m/data/naruto')
URI = 'bolt://127.0.0.1:7687'
AUTH = ('neo4j', 'ragm-local-dev')


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main() -> None:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as s:
        s.run('CREATE CONSTRAINT naruto_entity_name IF NOT EXISTS FOR (e:NarutoEntity) REQUIRE e.name IS UNIQUE')
        s.run('MATCH (e:NarutoEntity) DETACH DELETE e')
        nodes = load_jsonl(OUT / 'kg_nodes.jsonl')
        for i in range(0, len(nodes), 5000):
            s.run('UNWIND $rows AS row MERGE (e:NarutoEntity {name: row.name}) SET e.type = row.type, e.doc_id = row.doc_id', rows=nodes[i:i+5000])
        edges = load_jsonl(OUT / 'kg_edges.jsonl')
        for i in range(0, len(edges), 5000):
            s.run('UNWIND $rows AS row MATCH (a:NarutoEntity {name: row.a}), (b:NarutoEntity {name: row.b}) MERGE (a)-[r:MENTIONS]->(b) SET r.qtype = row.qtype, r.question = row.question', rows=edges[i:i+5000])
        n = s.run('MATCH (e:NarutoEntity) RETURN count(e) AS n').single()['n']
        r = s.run('MATCH ()-[r:MENTIONS]->() RETURN count(r) AS n').single()['n']
        print(f'naruto kg: {n} nodes, {r} edges')
    driver.close()


if __name__ == '__main__':
    main()