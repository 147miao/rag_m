"""把 kg_nodes.jsonl / kg_edges.jsonl 导入 Neo4j。"""
from __future__ import annotations

import json
import time
from pathlib import Path

from neo4j import GraphDatabase

OUT = Path("/run/media/myj/D/rag_m/data/rag")
URI = "bolt://127.0.0.1:7687"
AUTH = ("neo4j", "ragm-local-dev")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as s:
        s.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        s.run("CREATE INDEX entity_lang IF NOT EXISTS FOR (e:Entity) ON (e.lang)")
        t0 = time.perf_counter()
        nodes = load_jsonl(OUT / "kg_nodes.jsonl")
        for i in range(0, len(nodes), 5000):
            s.run("UNWIND $rows AS row MERGE (e:Entity {name: row.name}) SET e.lang = row.lang", rows=nodes[i : i + 5000])
        print(f"nodes: {len(nodes)} in {time.perf_counter()-t0:.0f}s", flush=True)
        t0 = time.perf_counter()
        edges = load_jsonl(OUT / "kg_edges.jsonl")
        for i in range(0, len(edges), 5000):
            s.run(
                "UNWIND $rows AS row "
                "MATCH (a:Entity {name: row.a}), (b:Entity {name: row.b}) "
                "CREATE (a)-[r:CO_EVIDENCE {qid: row.qid, qtype: row.qtype, question: row.question}]->(b)",
                rows=edges[i : i + 5000],
            )
            if (i // 5000) % 6 == 0:
                print(f"  edges {i+len(edges[i:i+5000])}/{len(edges)}", flush=True)
        print(f"edges: {len(edges)} in {time.perf_counter()-t0:.0f}s", flush=True)
        c = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        r = s.run("MATCH ()-[r:CO_EVIDENCE]->() RETURN count(r) AS n").single()["n"]
        print(f"neo4j: {c} nodes, {r} edges")
    driver.close()


if __name__ == "__main__":
    main()
