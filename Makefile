SHELL := /bin/bash
DATA_ROOT ?= /run/media/myj/D/rag_m
# 注意：Docker Desktop 只共享 $HOME，/run/media 下的 bind mount 会静默落到 VM 内部（host 看不到）。
# 因此容器卷放项目内 .volumes/（系统盘，体量小），D 盘只放数据集/模型/缓存（host 侧进程读写）。
export DOCKER_VOLUME_DIRECTORY ?= $(CURDIR)/.volumes
export HF_ENDPOINT ?= https://hf-mirror.com
export HF_HOME ?= $(DATA_ROOT)/hf_cache
export UV_PROJECT_ENVIRONMENT ?= .venv
UV ?= uv
COMPOSE ?= docker compose -f docker/compose.yaml

.PHONY: help mount-data check-data venv install up up-graph up-all down ps health api web test test-api fmt lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

DATA_DEVICE ?= /dev/nvme0n1p5

mount-data: ## 挂载 D 盘（udisksctl，无需 sudo；重启/拔出后需要重新挂载）
	@mountpoint -q $(DATA_ROOT) && echo "OK: $(DATA_ROOT) 已挂载" || (udisksctl mount -b $(DATA_DEVICE) && echo "mounted $(DATA_DEVICE)")

check-data: mount-data ## 校验 D 盘挂载与目录
	@mountpoint -q $(DATA_ROOT) && echo "OK: $(DATA_ROOT) 已挂载" || (echo "ERROR: D 盘未挂载，先执行 make mount-data"; exit 1)
	@mkdir -p $(DATA_ROOT)/data/raw $(DATA_ROOT)/data/processed $(DATA_ROOT)/data/corpus $(DATA_ROOT)/models $(DATA_ROOT)/hf_cache $(DATA_ROOT)/milvus_volumes $(DATA_ROOT)/neo4j_data
	@df -h $(DATA_ROOT) | tail -1

venv: ## 创建 Python 3.13 虚拟环境
	$(UV) venv --python 3.13 .venv

install: venv ## 安装依赖（含 CUDA torch）
	$(UV) sync --group dev

up: check-data ## 启动 Milvus（etcd+minio+standalone）
	$(COMPOSE) up -d etcd minio standalone

up-graph: check-data ## Milvus + Neo4j
	$(COMPOSE) --profile graph up -d

up-all: check-data ## Milvus + Neo4j + SearXNG
	$(COMPOSE) --profile graph --profile search up -d

down: ## 停止容器（保留数据）
	$(COMPOSE) --profile graph --profile search down

ps:
	$(COMPOSE) ps

health: ## 检查 Milvus 健康状态
	@curl -sf http://127.0.0.1:9091/healthz && echo || echo "Milvus 未就绪"

status: ## 打印实验进度看板（--watch 10 可实时刷新）
	$(UV) run python scripts/status.py $(if $(WATCH),--watch $(WATCH),)

use-all: ## 切换 API 到全量语料（演示+火影+论文）
	scripts/use_corpus.sh all

use-default: ## 切换 API 到默认演示语料（HotpotQA+CMRC）
	scripts/use_corpus.sh default

use-naruto: ## 切换 API 到火影语料
	scripts/use_corpus.sh naruto

use-papers: ## 切换 API 到 arXiv 论文语料
	scripts/use_corpus.sh papers

api: ## 启动 FastAPI（http://127.0.0.1:8000）
	$(UV) run uvicorn ragm.api.main:app --host 127.0.0.1 --port 8000

api-dev: ## 启动 FastAPI（热重载）
	$(UV) run uvicorn ragm.api.main:app --host 127.0.0.1 --port 8000 --reload

web: ## 启动前端 dev server（http://127.0.0.1:5173）
	cd web && pnpm dev

test:
	$(UV) run pytest -q

test-api: ## 只跑 API 契约测试
	$(UV) run pytest tests/test_api.py -q

fmt:
	$(UV) run ruff format src tests scripts

lint:
	$(UV) run ruff check src tests scripts

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
