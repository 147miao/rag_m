SHELL := /bin/bash
DATA_ROOT ?= /run/media/myj/D/rag_m
export DOCKER_VOLUME_DIRECTORY ?= $(DATA_ROOT)
export HF_ENDPOINT ?= https://hf-mirror.com
export HF_HOME ?= $(DATA_ROOT)/hf_cache
export UV_PROJECT_ENVIRONMENT ?= .venv
UV ?= uv
COMPOSE ?= docker compose -f docker/compose.yaml

.PHONY: help check-data venv install up up-graph up-all down ps health test fmt lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

check-data: ## 校验 D 盘挂载与目录
	@mountpoint -q $(DATA_ROOT)/.. && echo "OK: $(DATA_ROOT) 已挂载" || (echo "ERROR: D 盘未挂载"; exit 1)
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

test:
	$(UV) run pytest -q

fmt:
	$(UV) run ruff format src tests scripts

lint:
	$(UV) run ruff check src tests scripts

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
