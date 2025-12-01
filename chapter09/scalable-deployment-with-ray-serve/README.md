# Scalable deployment with Ray Serve

## 快速开始

### 拉取依赖

```bash
uv sync
```

### 构建索引

```bash
../.venv/bin/python src/build_index.py
```

### 启动服务

```bash

../.venv/bin/python src/serve_index.py
```

### 测试服务

```bash
uv run src/testbot.py
```

## 温馨提示
- facebook/faiss 官方仓库没有提供 pip 仓库包，因此用社区维护的 faiss-cpu 替换
