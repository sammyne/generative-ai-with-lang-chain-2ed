# Generative AI with LangChain 2ed

## 依赖
- VS Code >= 1.105.1
- uv >= 0.9

## 快速开始
```bash
uv sync
```

打开 jupyter notebook 文件后，参照 [Using uv with Jupyter / Using Jupyter from VS Code] 配置好 jupyter 使用的虚拟环境即可。

## 进度

- [x] 02. First Steps with LangChain
- [x] 03. Building Workflows with LangGraph
- [x] 04. Building Intelligent RAG Systems
- [ ] 05. Building Intelligent Agents
- [ ] 06. Advanced Applications and Multi-Agent Systems
- [ ] 07. Software Development and Data Analysis Agents
- [x] 08. Evaluation and Testing
- [ ] 09. Production-Ready LLM Deployment and Observability

## 温馨提示
- jupyter notebook 环境下，应使用 `uv pip install` 将依赖安装到 notebook 运行的虚拟环境下，并且保持 pyproject.toml 不改变。
  使用 `uv add` 安装依赖会导致 pyproject.toml 被更新。

## 参考文献
- 随书源码仓库 https://github.com/benman1/generative_ai_with_langchain/tree/second_edition
- https://reference.langchain.com/python/
- [阿里云百炼](https://bailian.console.aliyun.com)
- [Using uv with Jupyter / Using Jupyter from VS Code]

[Using uv with Jupyter / Using Jupyter from VS Code]: https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-within-a-project
