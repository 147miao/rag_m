"""Neo4j 实体图：实体链接、邻域扩展、图谱证据。"""
from __future__ import annotations

from neo4j import GraphDatabase

from ..api.schemas import GraphEdge, GraphNode, GraphPayload


class KGStore:
    def __init__(self, uri: str, user: str, password: str, label: str = "Entity", relation: str = "CO_EVIDENCE") -> None:
        labels = [x.strip() for x in label.split(",") if x.strip()]
        rels = [x.strip() for x in relation.split(",") if x.strip()]
        if len(rels) == 1 and len(labels) > 1:
            rels = rels * len(labels)
        self.specs = list(zip(labels, rels)) or [("Entity", "CO_EVIDENCE")]
        self.label = labels[0] if labels else "Entity"
        self.relation = rels[0] if rels else "CO_EVIDENCE"
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.names: list[str] = []
        self.name_set: set[str] = set()

    def load(self) -> int:
        with self.driver.session() as s:
            names: list[str] = []
            for label, _rel in self.specs:
                names.extend(r["n"] for r in s.run(f"MATCH (e:{label}) RETURN e.name AS n"))
            self.names = list(dict.fromkeys(names))
        self.name_set = set(self.names)
        return len(self.names)

    def close(self) -> None:
        self.driver.close()

    # ---------- 实体链接 ----------
    def link(self, query: str, seed_titles: list[str], max_entities: int = 6) -> list[str]:
        """实体链接：优先匹配问题里直接出现的实体名（长名优先），再用检索结果标题兜底。"""
        q = query.lower()
        matched: list[str] = []
        mentioned = [n for n in self.names if len(n) >= 4 and n.lower() in q]
        mentioned.sort(key=len, reverse=True)
        for n in mentioned:
            if n not in matched:
                matched.append(n)
                if len(matched) >= max_entities:
                    break
        for title in seed_titles:
            if len(matched) >= max_entities:
                break
            if title in self.name_set and title not in matched:
                matched.append(title)
        return matched[:max_entities]

    # ---------- 邻域扩展 ----------
    def expand(self, names: list[str], limit: int = 40) -> tuple[list[dict], list[str]]:
        if not names:
            return [], []
        rows: list[dict] = []
        with self.driver.session() as s:
            for label, rel in self.specs:
                cypher = (
                    f"MATCH (a:{label})-[r:{rel}]-(b:{label}) "
                    "WHERE a.name IN $names "
                    "RETURN a.name AS a, b.name AS b, r.qid AS qid, r.qtype AS qtype, r.question AS question "
                    "LIMIT $limit"
                )
                rows.extend(dict(rec) for rec in s.run(cypher, names=names, limit=limit))
                if len(rows) >= limit:
                    break
        neighbors: list[str] = []
        for r in rows:
            if r["b"] not in names and r["b"] not in neighbors:
                neighbors.append(r["b"])
        return rows, neighbors

    # ---------- 图数据（前端 Graph 面板） ----------
    def payload(self, question: str, seeds: list[str], edges: list[dict], doc_titles: list[str]) -> GraphPayload:
        nodes: list[GraphNode] = [GraphNode(id="q", label=question[:40], kind="query")]
        seen: set[str] = {"q"}
        for name in seeds:
            nid = "e:" + name
            if nid not in seen:
                seen.add(nid)
                nodes.append(GraphNode(id=nid, label=name[:40], kind="entity"))
        for e in edges:
            nid = "e:" + e["b"]
            if nid not in seen:
                seen.add(nid)
                nodes.append(GraphNode(id=nid, label=e["b"][:40], kind="entity"))
        for t in doc_titles:
            nid = "d:" + t
            if nid not in seen:
                seen.add(nid)
                nodes.append(GraphNode(id=nid, label=t[:40], kind="doc"))
        out_edges: list[GraphEdge] = []
        for name in seeds:
            out_edges.append(GraphEdge(from_="q", to="e:" + name, label="seed"))
        for e in edges:
            out_edges.append(GraphEdge(from_="e:" + e["a"], to="e:" + e["b"], label=e.get("qtype") or "co_evidence"))
        for t in doc_titles:
            out_edges.append(GraphEdge(from_="e:" + t, to="d:" + t, label="evidence"))
        return GraphPayload(nodes=nodes, edges=out_edges)