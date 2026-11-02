#!/usr/bin/env bash
# 切换 API 服务的知识库（systemd drop-in），用法：scripts/use_corpus.sh default|naruto|papers
set -euo pipefail
CORPUS="${1:-default}"
DROPIN_DIR="$HOME/.config/systemd/user/ragm-api.service.d"
DROPIN="$DROPIN_DIR/naruto.conf"
mkdir -p "$DROPIN_DIR"

case "$CORPUS" in
  naruto)
    cat > "$DROPIN" <<EOF
[Service]
Environment=MILVUS_COLLECTION=rag_naruto_chunks
Environment=MILVUS_CHILD_COLLECTION=rag_naruto_children
Environment=PARENTS_PATH=/run/media/myj/D/rag_m/data/naruto/parents.jsonl
Environment=KG_LABEL=NarutoEntity
Environment=KG_RELATION=MENTIONS
EOF
    ;;
  all)
    cat > "$DROPIN" <<EOF
[Service]
Environment=MILVUS_COLLECTION=rag_all_chunks
Environment=MILVUS_CHILD_COLLECTION=rag_children
Environment=PARENTS_PATH=/run/media/myj/D/rag_m/data/all/parents.jsonl
Environment=KG_LABEL=Entity,NarutoEntity
Environment=KG_RELATION=CO_EVIDENCE,MENTIONS
EOF
    ;;
  papers)
    cat > "$DROPIN" <<EOF
[Service]
Environment=MILVUS_COLLECTION=rag_papers_chunks
Environment=MILVUS_CHILD_COLLECTION=rag_papers_children
Environment=PARENTS_PATH=/run/media/myj/D/rag_m/data/papers/parents.jsonl
Environment=KG_LABEL=Entity
Environment=KG_RELATION=CO_EVIDENCE
EOF
    ;;
  default)
    rm -f "$DROPIN"
    ;;
  *)
    echo "usage: $0 default|naruto|papers|all" >&2
    exit 1
    ;;
esac

systemctl --user daemon-reload
systemctl --user restart ragm-api.service
sleep 3
echo "corpus=$CORPUS  (health: $(curl -s -m 5 http://127.0.0.1:8000/api/health | head -c 60)...)"