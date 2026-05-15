# LangGraph ReAct Agent Study

这个项目是一个用于学习、实验和持续升级的 LangGraph Agent 项目。它从手写 ReAct agent 起步，逐步沉淀工具系统、Skill 机制、测试体系、记忆能力、权限控制和工程化结构，目标是不断向更接近工业场景的自主智能体架构对齐。

当前阶段重点放在理解 LangGraph 的 `Node`、`State`、`Edge` 和 ReAct 循环，并把学习过程中的能力拆成可维护、可测试、可扩展的模块：

```text
START -> llm -> tools? -> llm -> ... -> END
```

其中：

- `State`: 使用自定义 `AgentState`，核心字段仍是 `messages`，同时显式记录工具审批状态和审计日志。
- `llm` node: 调用 OpenAI-compatible 接口，并让模型决定是否调用工具。
- `tool state machine`: 将工具流程拆成 `classify_tool_calls`、`approval_gate`、`execute_tools` 三个节点。
- `conditional edge`: 检查上一条 AI 消息里是否有 `tool_calls`。有就进入工具状态机，没有就结束。

## 项目结构

项目已按职责拆分，`react_agent.py` 只保留兼容入口，核心代码放在 `langraph_agent/` 包里：

```text
langraph_agent/
├── cli.py                 # 命令行参数、.env 加载、入口调度
├── config.py              # 默认模型、项目路径、超时和输出限制等配置
├── graph.py               # LangGraph ReAct 图、多轮对话、debug 输出
├── llm.py                 # ChatOpenAI / OpenAI-compatible 接口初始化
├── models.py              # 项目内共享数据结构
├── tool_guard.py          # 工具白名单、人工审核和执行器
├── skills/
│   └── registry.py        # Skill 发现、frontmatter 解析和目录定位
└── tools/
    ├── basic.py           # calculator、current_time 等基础工具
    ├── filesystem.py      # Deep Agents FilesystemBackend 文件工具
    ├── shell.py           # bash 工具和命令安全策略
    ├── skill_tools.py     # list_skills、load_skill 工具
    └── web_search.py      # Tavily web_search、web_extract 联网工具
```

后续扩展时建议：

- 新增普通工具：放到 `langraph_agent/tools/` 下的同类文件中，并在 `langraph_agent/tools/__init__.py` 注册到 `TOOLS`。
- 新增 Skill：继续放到项目根目录的 `skills/<skill-name>/SKILL.md`，不需要改 Python 代码。
- 修改模型或 API 默认值：优先改 `langraph_agent/config.py`，运行时差异继续用环境变量覆盖。
- 修改图结构或多轮记忆行为：集中改 `langraph_agent/graph.py`。

## 安装

推荐使用 `uv`：

```bash
uv sync
```

或者使用标准 venv：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 配置

不要把真实 AK 写进代码。复制环境变量模板：

```bash
cp .env.example .env
```

然后编辑 `.env`：

```bash
OPENAI_API_KEY=你的真实 AK
OPENAI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
OPENAI_MODEL=mimo-v2.5-pro
OPENAI_TEMPERATURE=0
TAVILY_API_KEY=你的 Tavily API Key
TAVILY_EXTRACT_CONTENT_LIMIT=12000
LANGRAPH_CHECKPOINT_DB_PATH=data/checkpoints.sqlite
LANGRAPH_COMPACT_TOKEN_THRESHOLD=8000
LANGRAPH_RECENT_MESSAGES_TO_KEEP=8
LANGRAPH_COMMAND_TIMEOUT_SECONDS=30
LANGRAPH_OUTPUT_LIMIT=8000
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的 LangSmith API Key
LANGSMITH_PROJECT=langraph-agent-dev
```

`OPENAI_*` 表示这里使用的是 OpenAI-compatible 协议配置，不绑定具体供应商。所有运行参数统一由 `langraph_agent/config.py` 中的 `config = Config()` 从环境变量加载并设置默认值，业务代码直接使用 `config.xxx`。thinking 开关统一放在 `config.OPENAI_EXTRA_BODY` 中，当前设置为 `{"thinking": {"type": "disabled"}}`，用于兼容 OpenAI-compatible Chat Completions + LangChain 工具调用链路。

LangSmith 当前只开启 tracing。项目启动时会通过 `python-dotenv` 加载 `.env`，只要 `.env` 中 `LANGSMITH_TRACING=true` 且 API Key 有效，LangGraph/LangChain 调用会自动上报 trace。

## 运行

```bash
python react_agent.py "北京现在几点？顺便算一下 23 * 47"
```

如果用 `uv`：

```bash
uv run python react_agent.py "北京现在几点？顺便算一下 23 * 47"
```

普通模式默认按 OpenAI SDK SSE 示例的 delta 风格流式打印回答：模型每返回一段内容就立即 `print(..., end="", flush=True)` 到控制台；`--debug` 也会显示 `[llm stream]` 实时正文，同时保留按 LangGraph 节点更新打印的学习视图。

需要查询互联网实时信息时，配置 `TAVILY_API_KEY` 后可以让模型自动调用联网工具：

- `web_search`: 用关键词搜索，返回搜索摘要和来源链接。这里的 `content` 是搜索结果片段，不是网页全文。
- `web_extract`: 用户已经给出 URL 时，提取该网页正文内容，适合总结、阅读和分析链接。

```bash
uv run python react_agent.py --debug "搜索 LangGraph 最新版本变化，并列出来源"
```

直接分析链接：

```bash
uv run python react_agent.py --debug "阅读这个链接并总结重点：https://example.com"
```

需要查看或修改项目文件时，Agent 会使用 Deep Agents 的 `FilesystemBackend` 文件工具，范围限制在当前项目根目录映射出的虚拟根路径 `/`：

- 自动执行：`ls`、`read_file`、`glob`、`grep` 等只读工具。
- 需要人工审核：`write_file`、`edit_file`、`bash`。
- 敏感路径如 `.env`、`.git/`、`.venv/` 即使通过只读工具访问，也会升级为人工审核。

遇到需要审核的工具调用时，CLI 会暂停并显示工具名、参数和原因。输入 `a` 或 `yes` 批准全部，输入 `r` 或 `no` 拒绝全部，也可以输入 `1,3` 只批准指定编号。

查看每一步图执行过程：

```bash
python react_agent.py --debug "北京现在几点？顺便算一下 23 * 47"
```

启动多轮对话：

```bash
python react_agent.py --chat
```

如果用 `uv`：

```bash
uv run python react_agent.py --chat
```

多轮对话也可以打开 debug，观察每一轮内部是否进入工具节点：

```bash
uv run python react_agent.py --chat --debug
```

debug 模式会按节点展示增量输出，并尽量拆分为：

- `reasoning`: 模型供应商如果返回思考/推理字段，则展示该字段。
- `tool_calls`: 模型请求调用的工具。
- `content`: 模型或工具返回的正文。

debug 模式下不会在末尾再次打印 `Final answer`，避免最终回答重复出现。

指定一个会话 ID：

```bash
uv run python react_agent.py --chat --thread-id study-session-1
```

多轮对话默认使用 SQLite checkpoint 持久化，数据库位置为 `data/checkpoints.sqlite`。同一个 `--thread-id` 会在程序重启后继续读取之前的对话状态。也可以用环境变量或 CLI 参数改数据库位置：

```bash
LANGRAPH_CHECKPOINT_DB_PATH=data/study.sqlite uv run python react_agent.py --chat --thread-id study-session-1
uv run python react_agent.py --chat --thread-id study-session-1 --checkpoint-db data/study.sqlite
```

退出多轮对话时输入：

```text
exit
```

查看当前扫描到的 Skill 元数据：

```bash
uv run python react_agent.py --list-skills
```

## 测试

项目使用 `pytest` 做自动化测试，测试代码放在 `tests/` 目录。当前测试主要覆盖：

- `calculator` 和安全算术解析。
- Skill frontmatter 解析、目录扫描和 catalog 格式化。
- `bash` 工具的危险命令拦截、白名单判断和基础执行路径。
- 文件工具注册、工具审批白名单、敏感路径升级审核。
- CLI 的 `--list-skills` 路径，避免依赖真实模型 API Key。

运行测试：

```bash
uv run pytest
```

## 关键代码

核心图结构在 `langraph_agent/graph.py` 的 `build_graph()`：

```python
builder = StateGraph(AgentState)
builder.add_node("llm", call_llm)
builder.add_node("classify_tool_calls", classify_tool_calls_node)
builder.add_node("approval_gate", approval_gate_node)
builder.add_node("execute_tools", execute_tools_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", _route_after_llm)
builder.add_conditional_edges("classify_tool_calls", _route_after_classify)
builder.add_edge("approval_gate", "execute_tools")
builder.add_edge("execute_tools", "llm")
graph = builder.compile()
```

ReAct 的关键点不是某个神秘框架 API，而是：

1. 模型先推理并决定是否需要工具。
2. 如果返回 `tool_calls`，条件边把状态路由到 `classify_tool_calls`。
3. `classify_tool_calls` 把调用分为自动批准和待人工审核，并写入 `pending_approvals`、`approved_tool_calls` 和 `tool_audit_log`。
4. 如果存在 `pending_approvals`，`approval_gate` 通过 `interrupt()` 暂停并等待用户逐项批准或拒绝。
5. `execute_tools` 执行已批准的工具，并为被拒绝的工具调用生成 `ToolMessage`。
6. 图回到 `llm` node，模型读取工具结果后继续推理。
7. 当模型不再返回 `tool_calls`，条件边路由到 `END`。

## 多轮对话

单次问答时，每次调用只传入当前用户问题：

```python
graph.invoke({"messages": [{"role": "user", "content": question}]})
```

多轮对话需要让图记住之前的 `messages`。这个项目默认使用 LangGraph 的 SQLite checkpointer，把状态保存到本地 SQLite 数据库：

```python
with sqlite_checkpointer("data/checkpoints.sqlite") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

每一轮调用时传入同一个 `thread_id`：

```python
config = {"configurable": {"thread_id": "default"}}
graph.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config=config,
)
```

这样你每一轮只需要传“新增的用户消息”。LangGraph 会根据 `thread_id` 取回上一轮保存的 State，再把新消息追加进去。

SQLite checkpoint 会跨进程保留同一个 `thread_id` 的历史。`build_graph(with_memory=True)` 仍保留 `MemorySaver` 路径，主要用于测试或临时内存实验；CLI 的 `run` 和 `chat` 路径默认使用 SQLite。

## 状态压缩

模型响应会返回 `usage_metadata.total_tokens`。项目会记录最近一次模型调用的真实 `total_tokens` 到 `last_total_tokens`，当最终回答后的 token 数超过 `LANGRAPH_COMPACT_TOKEN_THRESHOLD` 时，图会进入 `summarize_and_compact` 节点：

1. 用不绑定工具的 LLM 把旧消息合并进 `session_summary`。
2. 保留最近 `LANGRAPH_RECENT_MESSAGES_TO_KEEP` 条消息，并避免从孤立的 `ToolMessage` 开始。
3. 通过 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 物理裁剪 `messages` 状态。
4. 后续调用 LLM 时，会把 `session_summary` 注入 system prompt。

压缩只会发生在没有 `tool_calls` 的最终 AIMessage 之后，避免破坏 `AIMessage(tool_calls=...)` 和后续 `ToolMessage` 的配对关系。

默认配置：

```bash
LANGRAPH_COMPACT_TOKEN_THRESHOLD=8000
LANGRAPH_RECENT_MESSAGES_TO_KEEP=8
```

## 动态 Skill

项目内的 Skill 放在 `skills/` 目录，每个 Skill 是一个独立文件夹，必须包含 `SKILL.md`：

```text
skills/
├── langgraph-react-tutor/
│   └── SKILL.md
└── python-debug-helper/
    └── SKILL.md
```

`SKILL.md` 使用标准 Skill 格式，开头是 YAML frontmatter，并且至少包含 `name` 和 `description`：

```markdown
---
name: langgraph-react-tutor
description: Explain LangGraph Agent and ReAct mechanisms in Chinese...
---

# LangGraph ReAct Tutor

...
```

当前实现采用 progressive disclosure：

1. 启动图时扫描 `skills/*/SKILL.md`。
2. 只读取 YAML frontmatter 中的 `name` 和 `description`。
3. 把 Skill 元数据放入 system message，让模型判断是否需要某个 Skill。
4. 如果用户请求匹配某个 Skill，模型会调用 `load_skill(skill_name)`。
5. `load_skill` 再读取完整 `SKILL.md` 正文，并把说明作为工具结果返回给模型。

Skill 不是普通工具的替代品。它提供行为说明和工作流；需要计算、时间或其他外部能力时，Agent 仍然可以继续调用 `calculator`、`current_time` 等 tools。

可以通过环境变量切换 Skill 目录：

```bash
AGENT_SKILLS_DIR=/path/to/skills uv run python react_agent.py --list-skills
```

## 执行命令和 Skill 脚本

项目现在额外提供一个执行类工具：

- `bash(command)`: 执行本地 bash 命令。

`bash` 有三层策略：

1. 危险命令直接拦截，例如 `sudo`、递归强制删除、磁盘格式化、关机、管道下载脚本执行等。
2. 白名单命令直接执行，主要是只读或低风险命令，例如 `pwd`、`ls`、`rg`、`cat`、`git status`、`git diff`。
3. 不在白名单、也不在危险名单的命令，会通过 `input()` 要求用户确认后再执行。

Skill 脚本也使用 `bash` 执行。推荐流程：

1. 先通过 `load_skill(skill_name)` 读取 Skill 说明。
2. 根据 Skill 说明确认脚本路径。
3. 通过 `bash` 执行 `skills/<skill-name>/scripts/` 下的脚本。
4. 由于脚本执行不在白名单中，终端会通过 `input()` 要求用户确认。

示例 Skill 脚本：

```bash
uv run python react_agent.py --debug "使用 python-debug-helper 的 environment_report.py 脚本检查环境"
```

如果模型调用 `bash("uv run python skills/python-debug-helper/scripts/environment_report.py")`，终端会出现确认提示，输入 `y` 或 `yes` 才会真正执行。

## 注意

这个实现依赖模型支持 OpenAI 兼容的 tool calling。如果 `mimo-v2.5-pro` 没有按 OpenAI 格式返回 `tool_calls`，`tools_condition` 就不会进入工具节点。届时可以改成“文本 ReAct 格式 + 自定义 parser”的方案。
