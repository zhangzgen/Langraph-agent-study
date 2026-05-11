# LangGraph ReAct Agent Demo

这个项目演示如何用 LangGraph 手写一个 ReAct agent。你已经了解 `Node`、`State`、`Edge`，所以重点放在 ReAct 循环：

```text
START -> llm -> tools? -> llm -> ... -> END
```

其中：

- `State`: 使用 LangGraph 内置的 `MessagesState`，核心字段是 `messages`。
- `llm` node: 调用小米 OpenAI 兼容接口，并让模型决定是否调用工具。
- `tools` node: 使用 `ToolNode` 执行模型请求的工具调用。
- `conditional edge`: 使用 `tools_condition` 检查上一条 AI 消息里是否有 `tool_calls`。有就进入 `tools`，没有就结束。

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
XIAOMI_API_KEY=你的真实 AK
XIAOMI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
XIAOMI_MODEL=mimo-v2.5-pro
```

## 运行

```bash
python react_agent.py "北京现在几点？顺便算一下 23 * 47"
```

如果用 `uv`：

```bash
uv run python react_agent.py "北京现在几点？顺便算一下 23 * 47"
```

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

退出多轮对话时输入：

```text
exit
```

查看当前扫描到的 Skill 元数据：

```bash
uv run python react_agent.py --list-skills
```

## 关键代码

核心图结构在 `build_graph()`：

```python
builder = StateGraph(MessagesState)
builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")
graph = builder.compile()
```

ReAct 的关键点不是某个神秘框架 API，而是：

1. 模型先推理并决定是否需要工具。
2. 如果返回 `tool_calls`，条件边把状态路由到 `tools` node。
3. `ToolNode` 把工具结果追加为 `ToolMessage`。
4. 图回到 `llm` node，模型读取工具结果后继续推理。
5. 当模型不再返回 `tool_calls`，条件边路由到 `END`。

## 多轮对话

单次问答时，每次调用只传入当前用户问题：

```python
graph.invoke({"messages": [{"role": "user", "content": question}]})
```

多轮对话需要让图记住之前的 `messages`。这个项目使用 LangGraph 的 `MemorySaver` 作为 checkpointer：

```python
checkpointer = MemorySaver()
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

内存中的多轮历史只在当前进程内有效。程序退出后，`MemorySaver` 里的历史会消失。后续如果需要持久化，可以换成数据库型 checkpointer。

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
