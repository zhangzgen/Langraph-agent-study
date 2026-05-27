# LangGraph Agent Study

这个项目是一个基于 LangGraph 持续演进的 Agent 应用工程。项目从 ReAct 循环起步，但当前已不再只是基础调用示例，而是围绕真实交互场景搭建了可扩展、可审计、可持续运行的 Agent 执行链路。

当前项目已实现的核心能力包括：

- **多入口交互**：支持 CLI 单轮/多轮流式对话，并接入飞书机器人长连接与 CardKit 流式卡片回复。
- **计划与执行协作**：CLI 提供可选 plan 模式，Agent 可先澄清需求、读取上下文、生成执行计划，并在用户审核通过后进入实际执行。
- **受控工具调用**：集成文件读写、Shell、联网搜索、时间与计算等工具，将调用拆分为分类、审批和执行阶段；高风险操作支持 CLI 或飞书卡片人工审批与审计记录。
- **持久会话与上下文管理**：使用 SQLite 或 PostgreSQL checkpoint 保存多轮状态，支持中断后恢复执行；长上下文可自动压缩为摘要并继续注入后续对话。
- **动态 Skill 扩展**：运行时发现并加载 `skills/` 中的技能说明，使 Agent 能按任务选择专门工作流，而无需将所有能力硬编码进主流程。
- **提示词与工程化支撑**：提示词支持 LangSmith 远程优先、本地模板回退，并配套配置管理、测试覆盖和渠道侧状态持久化机制。

底层执行仍以 LangGraph 状态机为核心：主流程根据模型输出进入工具状态机或结束回答；启用计划模式时会在执行前插入计划生成与人工审核环节；渠道侧遇到敏感工具时可暂停当前 checkpoint，待审批后从同一会话继续运行。

## 项目结构

项目已按职责拆分，`react_agent.py` 只保留兼容入口，核心代码放在 `langraph_agent/` 包里：

```text
langraph_agent/
├── cli.py                 # 命令行参数与入口调度
├── config.py              # .env 加载、默认模型、项目路径与运行配置
├── checkpoints.py         # SQLite / PostgreSQL checkpoint 后端选择与生命周期
├── context.py             # token 统计、历史摘要与上下文物理裁剪
├── feishu_bot.py          # 飞书长连接事件、CardKit 流式卡片回复
├── feishu_approvals.py    # 飞书卡片审批映射、状态与按钮幂等持久化
├── graph.py               # LangGraph ReAct 图、plan/审批路由、多渠道流式输出
├── llm.py                 # ChatOpenAI / OpenAI-compatible 接口初始化
├── models.py              # 项目内共享数据结构
├── tool_guard.py          # 工具白名单、人工审核和执行器
├── prompt.py              # LangSmith 优先、本地回退的提示词加载与渲染
├── skills/
│   └── registry.py        # Skill 发现、frontmatter 解析和目录定位
└── tools/
    ├── basic.py           # calculator、current_time 等基础工具
    ├── filesystem.py      # Deep Agents FilesystemBackend 文件工具
    ├── planning.py        # plan 阶段 ask_human 澄清工具
    ├── shell.py           # bash 工具和命令安全策略
    ├── skill_tools.py     # list_skills、load_skill 工具
    └── web_search.py      # Tavily web_search、web_extract 联网工具
prompts/
├── react.txt              # 主执行 Agent 提示词本地回退模板
├── plan.txt               # plan 模式提示词本地模板
└── summary.txt            # 会话摘要提示词本地回退模板
```

后续扩展时建议：

- 新增普通工具：放到 `langraph_agent/tools/` 下的同类文件中，并在 `langraph_agent/tools/__init__.py` 注册到 `TOOLS`。
- 新增 Skill：继续放到项目根目录的 `skills/<skill-name>/SKILL.md`，不需要改 Python 代码。
- 修改模型或 API 默认值：优先改 `langraph_agent/config.py`，运行时差异继续用环境变量覆盖。
- 修改图结构或多轮记忆行为：集中改 `langraph_agent/graph.py`。

## 安装

项目要求 Python `>=3.11`。推荐使用 `uv` 安装运行依赖及开发依赖：

```bash
uv sync
```

或者使用标准 venv：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

安装后会提供 `react-agent` 与 `feishu-agent` 两个命令行入口；根目录的
`react_agent.py` 保留为兼容启动脚本。

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
OPENAI_THINKING_TYPE=disabled
TAVILY_API_KEY=你的 Tavily API Key
TAVILY_EXTRACT_CONTENT_LIMIT=12000
LANGRAPH_CHECKPOINT_DB_PATH=data/checkpoints.sqlite
LANGRAPH_CHECKPOINT_DATABASE_URL=postgresql://langraph_agent:dev_password@localhost:5432/langraph_agent
LANGRAPH_COMPACT_TOKEN_THRESHOLD=8000
LANGRAPH_RECENT_MESSAGES_TO_KEEP=8
LANGRAPH_COMMAND_TIMEOUT_SECONDS=30
LANGRAPH_OUTPUT_LIMIT=8000
FEISHU_APP_ID=你的飞书应用 App ID
FEISHU_APP_SECRET=你的飞书应用 App Secret
FEISHU_BASE_URL=https://open.feishu.cn
FEISHU_CARD_UPDATE_INTERVAL_MS=250
FEISHU_WORKER_COUNT=4
FEISHU_APPROVAL_DATABASE_URL=postgresql://langraph_agent:dev_password@localhost:5432/langraph_agent
FEISHU_APPROVAL_DB_PATH=data/feishu_approvals.sqlite
FEISHU_APPROVAL_POOL_MIN_SIZE=1
FEISHU_APPROVAL_POOL_MAX_SIZE=5
FEISHU_APPROVAL_POOL_TIMEOUT_SECONDS=3
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的 LangSmith API Key
LANGSMITH_PROJECT=langraph-agent-dev
LANGRAPH_REACT_PROMPT_ID=langraph-agent-react-system
LANGRAPH_PLAN_PROMPT_ID=
LANGRAPH_SUMMARY_PROMPT_ID=langraph-agent-summary
```

`OPENAI_*` 表示这里使用的是 OpenAI-compatible 协议配置，不绑定具体供应商。项目参数主要由 `langraph_agent/config.py` 中的 `config = Config()` 从环境变量加载，业务代码直接使用 `config.xxx`。`OPENAI_THINKING_TYPE` 会组装到 `config.OPENAI_EXTRA_BODY={"thinking": {"type": ...}}`，默认值为 `disabled`，用于兼容当前 OpenAI-compatible Chat Completions + LangChain 工具调用链路。

checkpoint 后端根据连接串是否为空选择：`LANGRAPH_CHECKPOINT_DATABASE_URL` 非空时使用 PostgreSQL，否则使用 `LANGRAPH_CHECKPOINT_DB_PATH` 指定的 SQLite 文件。`.env.example` 中提供的是 PostgreSQL 示例连接串；本地没有 PostgreSQL 服务时应显式留空：

```bash
LANGRAPH_CHECKPOINT_DATABASE_URL=
FEISHU_APPROVAL_DATABASE_URL=
```

常用可选配置还包括：`LANGRAPH_PROJECT_ROOT` 用于修改文件工具的项目根目录，`AGENT_SKILLS_DIR` 用于切换 Skill 扫描目录。测试或临时实验可把 `LANGRAPH_CHECKPOINT_DB_PATH` 设置为 `:memory:` 选择内存 SQLite；`LANGRAPH_SQLITE_IN_MEMORY` 可在需要时修改该标识值。

项目启动时会通过 `python-dotenv` 加载 `.env`。只要 `.env` 中 `LANGSMITH_TRACING=true` 且 API Key 有效，LangGraph/LangChain 调用会自动上报 trace。

提示词统一由 `langraph_agent/prompt.py` 加载，文本回退模板维护在项目根目录的 `prompts/` 中。配置对应的 `LANGRAPH_*_PROMPT_ID` 时，会优先使用 LangSmith 模板；远程模板无法加载时回退到同名本地模板。`plan` 默认不配置远程标识，因此保持从 `prompts/plan.txt` 加载；需要远程维护时设置 `LANGRAPH_PLAN_PROMPT_ID` 即可。

## 运行

```bash
python react_agent.py "北京现在几点？顺便算一下 23 * 47"
```

如果用 `uv`：

```bash
uv run python react_agent.py "北京现在几点？顺便算一下 23 * 47"
```

也可以使用安装生成的命令入口：

```bash
uv run react-agent "北京现在几点？顺便算一下 23 * 47"
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

需要先制定执行计划并由用户审核时，可以启用 plan 模式：

```bash
uv run python react_agent.py --plan "给当前项目增加一个新功能"
```

plan 模式会先进入一个前置 ReAct 循环：模型可以使用 `ask_human` 向用户提出选择题或说明题，也可以使用 `ls`、`read_file`、`glob`、`grep` 读取项目文件。敏感路径仍沿用工具审批规则。`ask_human` 固定接收 `{"choose_list":{"问题一":["选项 A","选项 B"],"问题二":["选项 C","选项 D"]}}` 或 `{"question":"需要补充说明的问题"}` 两种格式，选择题可在一次调用中包含多个问题，工具会格式化展示并通过中断恢复收集终端输入。`ask_human` 参数校验异常会作为工具结果反馈给计划模型重新处理，而不会终止图执行。模型确认信息足够后会流式输出“执行计划书”，CLI 随后暂停审核；直接回车或输入 `yes` 通过，其他输入会作为调整意见返回给计划模型重写计划书。计划通过后，同一个 `AgentState` 会继续路由到主执行 ReAct。

`--chat --plan` 会对多轮会话中的每一轮新任务执行上述计划审核流程；飞书机器人入口当前直接运行主执行图，不启用 CLI 的 plan 交互。

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

未配置 PostgreSQL 连接串时，多轮对话使用 SQLite checkpoint 持久化，数据库位置为 `data/checkpoints.sqlite`。同一个 `--thread-id` 会在程序重启后继续读取之前的对话状态。也可以用环境变量或 CLI 参数改 SQLite 数据库位置：

```bash
LANGRAPH_CHECKPOINT_DB_PATH=data/study.sqlite uv run python react_agent.py --chat --thread-id study-session-1
uv run python react_agent.py --chat --thread-id study-session-1 --checkpoint-db data/study.sqlite
```

如果已经用 Docker 启动 PostgreSQL，可以改用 PostgreSQL checkpoint。项目会直接从 `.env` 读取 `LANGRAPH_CHECKPOINT_DATABASE_URL`；该值非空时优先使用 PostgreSQL，留空时回退 SQLite：

```bash
LANGRAPH_CHECKPOINT_DATABASE_URL=postgresql://langraph_agent:dev_password@localhost:5432/langraph_agent
uv run python react_agent.py --chat --thread-id study-session-1
```

配置了 PostgreSQL 时，`--checkpoint-db` 不会覆盖 PostgreSQL 后端；它只在 SQLite 模式下生效。单轮 `run` 也会创建带随机 `thread_id` 的 checkpoint，以支持审批中断后的恢复，但不会作为命名会话供后续对话继续使用。

退出多轮对话时输入：

```text
exit
```

## 飞书应用机器人

项目通过飞书官方 Python SDK 的长连接订阅事件，不需要部署公网回调地址。用户在工作台打开机器人单聊并发送文本后，处理链路如下：

1. 长连接接收 `im.message.receive_v1` 与 `p2.card.action.trigger`，回调立即把任务投递到后台线程。
2. `chat_id` 映射为 LangGraph `thread_id`（格式为 `feishu:<chat_id>`），因此同一单聊自动复用 checkpoint 多轮历史。
3. 机器人创建启用了 `streaming_mode` 的 CardKit JSON 2.0 卡片实体，并以卡片消息发送给当前会话。
4. 模型的流式 token 按 `FEISHU_CARD_UPDATE_INTERVAL_MS` 节流更新当前 markdown 内容块；工具调用会按发生位置插入卡片时间线，`bash` 命令使用代码块展示，但不展示工具执行结果。
5. 需要人工审批时，原卡片关闭 `streaming_mode`，在对应工具块中用分隔线和横向按钮展示当前审批项；多项全部决定后才使用 `Command(resume=...)` 恢复原 checkpoint。
6. 卡片阶段始终在同一消息内切换为“生成中 -> 待审批 -> 执行中 -> 完成”。映射、审批状态、CardKit `sequence` 和按钮幂等键默认通过 PostgreSQL 连接池存储；未配置或启动时无法连接 PostgreSQL 时回退到 `FEISHU_APPROVAL_DB_PATH` 指定的 SQLite 文件。

在飞书开放平台应用后台完成以下配置：

- 将事件订阅的接收方式设置为“使用长连接接收事件”。
- 订阅事件 `im.message.receive_v1` 与 `p2.card.action.trigger`。已开通的 `im.chat.access_event.bot_p2p_chat_entered_v1` 和 `p2p_chat_create` 可保留，但回答流程不依赖它们。
- 开通 `im:message:send_as_bot`、`im:message`、`cardkit:card:write` 权限，并发布可供测试用户使用的应用版本。

把应用凭证写入本地 `.env`，不要提交凭证到 Git：

```bash
FEISHU_APP_ID=cli_xxxxxxxxxxxxxxxx
FEISHU_APP_SECRET=你的应用密钥
FEISHU_APPROVAL_DATABASE_URL=postgresql://langraph_agent:dev_password@localhost:5432/langraph_agent
FEISHU_APPROVAL_DB_PATH=data/feishu_approvals.sqlite
FEISHU_APPROVAL_POOL_MIN_SIZE=1
FEISHU_APPROVAL_POOL_MAX_SIZE=5
FEISHU_APPROVAL_POOL_TIMEOUT_SECONDS=3
```

未单独设置 `FEISHU_APPROVAL_DATABASE_URL` 时，审批存储会沿用 `LANGRAPH_CHECKPOINT_DATABASE_URL`；将其显式留空可强制使用 SQLite。PostgreSQL 池在机器人启动时预热，默认最大连接数为 `FEISHU_WORKER_COUNT + 1`，为后台回答线程和卡片 action 回调预留连接。只有启动连接失败会回退 SQLite；服务运行后不会在一次审批会话中切换数据库，以保持按钮幂等事务的一致性。

安装依赖并启动机器人：

```bash
uv sync
uv run feishu-agent
```

当前飞书入口只接受单聊文本消息发起回答；高风险工具审批通过同一张回答卡片中的按钮完成。审批暂停期间，同一聊天的新问题会收到处理中提示，防止向暂停中的 `feishu:<chat_id>` checkpoint 并行追加轮次。

查看当前扫描到的 Skill 元数据：

```bash
uv run python react_agent.py --list-skills
```

## 测试

项目使用 `pytest` 做自动化测试，测试代码放在 `tests/` 目录。当前测试主要覆盖：

- `calculator` 和安全算术解析。
- Skill frontmatter 解析、目录扫描和 catalog 格式化。
- `bash` 工具的危险命令拦截、直接执行白名单和确认路径。
- 文件工具注册、工具审批分类、敏感路径升级审核和图流式输出。
- SQLite/PostgreSQL checkpoint 选择、连接串隐藏及上下文摘要压缩。
- plan 模式的 `ask_human`、计划审核与恢复执行流程。
- Tavily 工具格式化、提示词回退加载与模型初始化配置。
- 飞书 CardKit 流式回复、逐项审批、重启恢复和 PostgreSQL 到 SQLite 回退。
- CLI 参数透传与 `--list-skills` 无 API Key 执行路径。

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
builder.add_node("summarize_and_compact", summarize_and_compact)
if plan_mode:
    builder.add_node("plan_llm", call_plan_llm)
    builder.add_node("execute_plan_tools", execute_plan_tools)
    builder.add_node("review_plan", review_plan)
    builder.add_edge(START, "plan_llm")
    builder.add_conditional_edges(
        "plan_llm",
        lambda state: "execute_plan_tools" if has_tool_calls(state) else "review_plan",
    )
    builder.add_edge("execute_plan_tools", "plan_llm")
    builder.add_conditional_edges(
        "review_plan",
        lambda state: "llm" if state.get("plan_approved") else "plan_llm",
    )
else:
    builder.add_edge(START, "llm")
builder.add_conditional_edges(
    "llm",
    lambda state: _route_after_llm(
        state,
        compact_token_threshold=compact_token_threshold,
        recent_messages_to_keep=recent_messages_to_keep,
    ),
)
builder.add_conditional_edges("classify_tool_calls", _route_after_classify)
builder.add_edge("approval_gate", "execute_tools")
builder.add_edge("execute_tools", "llm")
builder.add_edge("summarize_and_compact", END)
graph = builder.compile()
```

ReAct 的关键点不是某个神秘框架 API，而是：

1. 启用 plan 模式时，图先让计划模型使用 `ask_human` 或只读文件工具补充上下文，并等待用户审核计划书；否则直接进入主执行模型。
2. 主执行模型推理并决定是否需要工具。
3. 如果返回 `tool_calls`，条件边把状态路由到 `classify_tool_calls`。
4. `classify_tool_calls` 把调用分为自动批准和待人工审核，并写入 `pending_approvals`、`approved_tool_calls` 和 `tool_audit_log`。
5. 如果存在 `pending_approvals`，`approval_gate` 通过 `interrupt()` 暂停并等待用户逐项批准或拒绝。
6. `execute_tools` 执行已批准的工具，并为被拒绝的工具调用生成 `ToolMessage`。
7. 图回到 `llm` node，模型读取工具结果后继续推理。
8. 当最终回答达到压缩阈值且历史足够长时，图进入 `summarize_and_compact`；否则路由到 `END`。

## 多轮对话

单次问答时，每次调用只传入当前用户问题：

```python
graph.invoke({"messages": [{"role": "user", "content": question}]})
```

多轮对话需要让图记住之前的 `messages`。这个项目默认使用 LangGraph 的 SQLite checkpointer，把状态保存到本地 SQLite 数据库；配置 PostgreSQL URL 后会改用 PostgreSQL：

```python
with checkpoint_saver() as checkpointer:
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

SQLite 和 PostgreSQL checkpoint 都会跨进程保留同一个 `thread_id` 的历史。`build_graph(with_memory=True)` 仍保留 `MemorySaver` 路径，主要用于测试或临时内存实验；CLI 的 `run` 和 `chat` 路径会读取 `config.CHECKPOINT_DATABASE_URL`，该配置非空时优先使用 PostgreSQL，否则使用 SQLite。

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
│   ├── SKILL.md
│   └── agents/openai.yaml
└── python-debug-helper/
    ├── SKILL.md
    ├── agents/openai.yaml
    └── scripts/environment_report.py
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

`bash` 有两层审批与拦截策略：

1. 在 Agent 主执行图中，所有 `bash` 调用都属于 `REVIEW_REQUIRED_TOOLS`，会先由 `approval_gate` 中断并要求用户批准；批准后同一次调用不会再次弹出重复确认。
2. `bash` 工具自身仍保留防护：危险命令始终直接拦截，例如 `sudo`、递归强制删除、磁盘格式化、关机、管道下载脚本执行等；工具被独立调用而未带图层批准上下文时，只有 `pwd`、`ls`、`rg`、`git status`、`git diff` 等低风险命令可以直接执行，其余命令通过 `input()` 确认。

Skill 脚本也使用 `bash` 执行。推荐流程：

1. 先通过 `load_skill(skill_name)` 读取 Skill 说明。
2. 根据 Skill 说明确认脚本路径。
3. 通过 `bash` 执行 `skills/<skill-name>/scripts/` 下的脚本。
4. 主执行图会把该 `bash` 请求交给人工审批；用户批准后才会真正运行脚本。

示例 Skill 脚本：

```bash
uv run python react_agent.py --debug "使用 python-debug-helper 的 environment_report.py 脚本检查环境"
```

如果模型调用 `bash("uv run python skills/python-debug-helper/scripts/environment_report.py")`，CLI 会把该工具调用列为待审核项；批准对应项后才会真正执行。

## 注意

这个实现依赖模型支持 OpenAI 兼容的 tool calling。如果 `mimo-v2.5-pro` 没有按 OpenAI 格式返回 `tool_calls`，`tools_condition` 就不会进入工具节点。届时可以改成“文本 ReAct 格式 + 自定义 parser”的方案。
