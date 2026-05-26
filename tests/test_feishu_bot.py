from __future__ import annotations

import json
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage
from langgraph.types import Command

from langraph_agent.feishu_approvals import FeishuApprovalStore
from langraph_agent.feishu_bot import CardStreamWriter, FeishuBotService, FeishuOpenAPIClient
from langraph_agent.graph import ChannelRunOutcome


def test_feishu_open_api_client_sends_cardkit_requests() -> None:
    """验证 CardKit 流式回答使用的开放平台协议。

    Description:
        通过 MockTransport 捕获认证、创建卡片、发送卡片、更新文本及完成配置的
        请求，确保请求路径和关键请求体符合卡片实体链路。
    Args:
        无。
    Returns:
        None: 测试只通过断言校验请求数据。
    """
    requests: list[tuple[str, str, dict]] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        """返回飞书接口的模拟成功响应。

        Description:
            记录每个请求的 HTTP 方法、路径和 JSON 请求体，并为认证和卡片创建
            返回对应模拟字段。
        Args:
            request (httpx.Request): httpx MockTransport 收到的请求。
        Returns:
            httpx.Response: 符合飞书业务成功格式的模拟响应。
        """
        payload = json.loads(request.content.decode("utf-8"))
        requests.append((request.method, request.url.path, payload))
        if request.url.path.endswith("/tenant_access_token/internal"):
            return httpx.Response(
                200,
                json={"code": 0, "tenant_access_token": "token", "expire": 7200},
            )
        if request.url.path.endswith("/cardkit/v1/cards"):
            return httpx.Response(200, json={"code": 0, "data": {"card_id": "card-1"}})
        return httpx.Response(200, json={"code": 0, "data": {}})

    client = FeishuOpenAPIClient(
        "app-id",
        "app-secret",
        transport=httpx.MockTransport(handle_request),
    )
    try:
        card_id = client.create_streaming_card()
        client.send_card("chat-1", card_id)
        client.update_card_text(card_id, "hello", 1)
        client.replace_card(
            card_id,
            {"schema": "2.0", "body": {"elements": []}},
            2,
        )
        client.finish_streaming_card(card_id, 3)
    finally:
        client.close()

    assert card_id == "card-1"
    assert [path for _, path, _ in requests].count(
        "/open-apis/auth/v3/tenant_access_token/internal"
    ) == 1
    create_body = requests[1][2]
    card = json.loads(create_body["data"])
    assert card["schema"] == "2.0"
    assert card["config"]["streaming_mode"] is True
    assert card["body"]["elements"][0]["element_id"] == "status"
    assert card["body"]["elements"][1]["element_id"] == "answer"
    send_body = requests[2][2]
    assert send_body["msg_type"] == "interactive"
    assert json.loads(send_body["content"])["data"]["card_id"] == "card-1"
    assert requests[3][2]["content"] == "hello"
    assert requests[4][1] == "/open-apis/cardkit/v1/cards/card-1"
    assert requests[4][2]["sequence"] == 2
    assert json.loads(requests[4][2]["card"]["data"])["schema"] == "2.0"
    assert requests[5][2]["sequence"] == 3


class FakeFeishuClient:
    """记录机器人服务对飞书 API 的调用。"""

    def __init__(self) -> None:
        """初始化测试调用记录容器。

        Description:
            创建用于保存发送与更新行为的可变列表。
        Args:
            无。
        Returns:
            None: 该方法仅初始化记录状态。
        """
        self.calls: list[tuple] = []

    def create_streaming_card(self) -> str:
        """返回固定测试卡片 ID。

        Description:
            记录创建卡片动作并避免调用真实飞书接口。
        Args:
            无。
        Returns:
            str: 固定的模拟卡片 ID。
        """
        self.calls.append(("create",))
        return "card-1"

    def send_card(self, chat_id: str, card_id: str) -> None:
        """记录卡片发送动作。

        Description:
            保存机器人尝试发送卡片的会话与卡片标识。
        Args:
            chat_id (str): 目标会话 ID。
            card_id (str): 目标卡片 ID。
        Returns:
            None: 该方法只记录调用。
        """
        self.calls.append(("send_card", chat_id, card_id))

    def update_card_text(
        self,
        card_id: str,
        text: str,
        sequence: int,
        element_id: str = "answer",
    ) -> None:
        """记录卡片正文更新动作。

        Description:
            保存模拟流式写入时的文本与序号。
        Args:
            card_id (str): 目标卡片 ID。
            text (str): 完整回答正文。
            sequence (int): 更新序号。
            element_id (str): 当前更新的 markdown 内容块 ID。
        Returns:
            None: 该方法只记录调用。
        """
        self.calls.append(("update", card_id, text, sequence, element_id))

    def finish_streaming_card(self, card_id: str, sequence: int) -> None:
        """记录卡片流式完成动作。

        Description:
            保存关闭 streaming_mode 所用的卡片 ID 与序号。
        Args:
            card_id (str): 目标卡片 ID。
            sequence (int): 完成更新序号。
        Returns:
            None: 该方法只记录调用。
        """
        self.calls.append(("finish", card_id, sequence))

    def replace_card(self, card_id: str, card: dict, sequence: int) -> None:
        """记录整卡覆盖动作。

        Description:
            保存交互状态切换时生成的 Card JSON 与更新序号。
        Args:
            card_id (str): 目标卡片 ID。
            card (dict): 待展示的完整卡片数据。
            sequence (int): 更新序号。
        Returns:
            None: 该方法只记录调用。
        """
        self.calls.append(("replace", card_id, card, sequence))

    def send_text(self, chat_id: str, text: str) -> None:
        """记录普通文本发送动作。

        Description:
            支持测试异常路径需要的文本提示接口。
        Args:
            chat_id (str): 目标会话 ID。
            text (str): 文本内容。
        Returns:
            None: 该方法只记录调用。
        """
        self.calls.append(("send_text", chat_id, text))

    def close(self) -> None:
        """模拟关闭客户端。

        Description:
            提供与正式客户端相同的资源释放接口。
        Args:
            无。
        Returns:
            None: 该测试替身没有资源需要释放。
        """
        self.calls.append(("close",))


def test_bot_service_maps_chat_to_thread_and_finishes_card() -> None:
    """验证机器人服务按飞书会话保持多轮记忆。

    Description:
        直接执行一轮后台回答，确认传给 LangGraph 的 thread_id 从 chat_id
        稳定派生，并在文本输出完成后关闭卡片流式状态。
    Args:
        无。
    Returns:
        None: 测试通过断言验证行为。
    """
    client = FakeFeishuClient()
    stream_calls: list[tuple[str, str, str | None]] = []

    def fake_answer_streamer(
        question: str,
        thread_id: str,
        on_text,
        checkpoint_db_path: str | None,
    ) -> AIMessage:
        """模拟 LangGraph 的流式渠道执行。

        Description:
            记录收到的输入与会话标识，然后分两次回调累计文本。
        Args:
            question (str): 测试用户问题。
            thread_id (str): 机器人生成的 checkpoint 会话标识。
            on_text (Callable): 卡片文本更新回调。
            checkpoint_db_path (str | None): SQLite 路径覆盖值。
        Returns:
            AIMessage: 固定的最终回答消息。
        """
        stream_calls.append((question, thread_id, checkpoint_db_path))
        on_text("你")
        on_text("你好")
        return AIMessage(content="你好")

    service = FeishuBotService(
        client,
        answer_streamer=fake_answer_streamer,
        update_interval_ms=0,
    )
    service._answer_message("oc_chat", "你好")
    service.close()

    assert stream_calls == [("你好", "feishu:oc_chat", None)]
    assert ("send_card", "oc_chat", "card-1") in client.calls
    assert ("update", "card-1", "你", 1, "answer") in client.calls
    assert ("update", "card-1", "你好", 2, "answer") in client.calls
    assert ("finish", "card-1", 3) in client.calls


class FakeInteractiveRunner:
    """模拟先产生两项审批、再按聚合决定恢复完成的渠道运行器。"""

    def __init__(self) -> None:
        """初始化运行输入记录。

        Description:
            创建用于断言仅在全部审批完成后恢复图的输入列表。
        Args:
            无。
        Returns:
            None: 该方法仅初始化记录状态。
        """
        self.inputs: list[str | Command] = []

    def __call__(
        self,
        next_input: str | Command,
        thread_id: str,
        on_text,
        on_tool_calls,
        checkpoint_db_path: str | None,
    ) -> ChannelRunOutcome:
        """返回固定的中断或完成结果。

        Description:
            首次调用上报两个需审批工具并中断；恢复调用输出最终正文并完成。
        Args:
            next_input (str | Command): 用户问题或恢复命令。
            thread_id (str): 当前 checkpoint thread 标识。
            on_text (Callable): 模型文本更新回调。
            on_tool_calls (Callable): 工具概要更新回调。
            checkpoint_db_path (str | None): 测试未使用的 checkpoint 路径。
        Returns:
            ChannelRunOutcome: 首次为审批中断，第二次为已完成结果。
        """
        assert thread_id == "feishu:oc_chat"
        assert checkpoint_db_path is None
        self.inputs.append(next_input)
        if len(self.inputs) == 1:
            on_text("开始处理。")
            tool_calls = [
                {
                    "id": "call_write",
                    "name": "write_file",
                    "approval_required": True,
                    "display_content": "目标路径：`notes.txt`",
                },
                {
                    "id": "call_bash",
                    "name": "bash",
                    "approval_required": True,
                    "display_content": "执行命令：\n```bash\npytest -q\n```",
                },
            ]
            on_tool_calls(tool_calls)
            return ChannelRunOutcome(
                final_message=None,
                interrupt_request={
                    "type": "tool_approval",
                    "tool_calls": [
                        {"id": "call_write", "name": "write_file"},
                        {"id": "call_bash", "name": "bash"},
                    ],
                },
            )
        assert isinstance(next_input, Command)
        assert next_input.resume == {"approved_call_ids": ["call_write"]}
        on_text("处理")
        on_text("处理完成")
        return ChannelRunOutcome(
            final_message=AIMessage(content="处理完成"),
            interrupt_request=None,
        )


def test_interactive_card_approves_tools_one_by_one_then_resumes(
    tmp_path: Path,
) -> None:
    """验证卡片逐项审批并在全部决定后恢复同一会话。

    Description:
        模拟两个需审核工具，断言第一次按钮动作只切换下一项，重复动作被忽略，
        并模拟最后一项决定落库后由重投动作恢复 LangGraph 并完成原卡片。
    Args:
        tmp_path (Path): pytest 提供的临时审批数据库目录。
    Returns:
        None: 测试只通过断言校验持久化状态与卡片内容。
    """
    client = FakeFeishuClient()
    runner = FakeInteractiveRunner()
    store = FeishuApprovalStore(tmp_path / "approvals.sqlite")
    service = FeishuBotService(
        client,
        agent_runner=runner,
        approval_store=store,
        update_interval_ms=0,
    )
    service._answer_message("oc_chat", "请修改文件并执行命令")

    session = store.get_session("card-1")
    assert session["status"] == "pending_approval"
    pending_card = [call for call in client.calls if call[0] == "replace"][-1][2]
    assert pending_card["config"]["streaming_mode"] is False
    pending_elements = pending_card["body"]["elements"]
    assert (
        pending_elements[5]["columns"][0]["elements"][0]["behaviors"][0]["value"][
            "tool_call_id"
        ]
        == "call_write"
    )
    assert pending_elements[1]["content"] == "开始处理。"
    assert pending_elements[2]["tag"] == "hr"
    assert pending_elements[5]["tag"] == "column_set"
    assert [column["elements"][0]["text"]["content"] for column in pending_elements[5]["columns"]] == [
        "批准",
        "拒绝",
    ]
    assert "```bash\npytest -q\n```" in pending_elements[7]["content"]
    assert all(element["tag"] != "action" for element in pending_elements)

    service._handle_tool_action(
        {
            "action": "approve_tool",
            "card_id": "card-1",
            "tool_call_id": "call_write",
            "action_key": "card-1:call_write:approve",
        }
    )
    service._handle_tool_action(
        {
            "action": "approve_tool",
            "card_id": "card-1",
            "tool_call_id": "call_write",
            "action_key": "card-1:call_write:approve",
        }
    )
    assert len(runner.inputs) == 1
    second_card = [call for call in client.calls if call[0] == "replace"][-1][2]
    second_columns = next(
        element for element in second_card["body"]["elements"] if element["tag"] == "column_set"
    )
    assert (
        second_columns["columns"][0]["elements"][0]["behaviors"][0]["value"]["tool_call_id"]
        == "call_bash"
    )

    assert store.decide_tool(
        "card-1",
        "call_bash",
        approved=False,
        action_key="card-1:call_bash:reject",
    )
    service.close()
    restarted_service = FeishuBotService(
        client,
        agent_runner=runner,
        approval_store=store,
        update_interval_ms=0,
    )
    restarted_service.close()

    completed = store.get_session("card-1")
    final_card = [call for call in client.calls if call[0] == "replace"][-1][2]
    final_content = "\n".join(
        element.get("content", "") for element in final_card["body"]["elements"]
    )
    assert completed["status"] == "completed"
    assert len(runner.inputs) == 2
    assert final_card["config"]["streaming_mode"] is False
    assert "开始处理。" in final_content
    assert "**工具调用：`write_file`** (已批准)" in final_content
    assert "**工具调用：`bash`** (已拒绝)" in final_content
    assert "```bash\npytest -q\n```" in final_content
    assert final_card["body"]["elements"][-1]["content"] == "处理完成"


def test_restart_fails_abandoned_active_sessions_and_unlocks_chat(
    tmp_path: Path,
) -> None:
    """验证服务重启会终止没有恢复入口的活动卡片。

    Description:
        预置生成中和无审批恢复入口的执行中会话，初始化新服务后确认两者被
        更新为失败卡片，且聊天不再被活动会话查询拦截。
    Args:
        tmp_path (Path): pytest 提供的临时审批数据库目录。
    Returns:
        None: 测试通过断言校验持久化状态与卡片刷新内容。
    """
    client = FakeFeishuClient()
    runner = FakeInteractiveRunner()
    store = FeishuApprovalStore(tmp_path / "approvals.sqlite")
    store.create_session("card-generating", "oc_generating", "feishu:oc_generating")
    store.create_session("card-executing", "oc_executing", "feishu:oc_executing")
    store.set_status("card-executing", "executing")

    restarted_service = FeishuBotService(
        client,
        agent_runner=runner,
        approval_store=store,
        update_interval_ms=0,
    )
    restarted_service.close()

    assert store.get_session("card-generating")["status"] == "failed"
    assert store.get_session("card-executing")["status"] == "failed"
    assert store.find_active_session("oc_generating") is None
    assert store.find_active_session("oc_executing") is None
    replaced_cards = {
        call[1]: call[2] for call in client.calls if call[0] == "replace"
    }
    for card_id in {"card-generating", "card-executing"}:
        elements = replaced_cards[card_id]["body"]["elements"]
        assert replaced_cards[card_id]["config"]["streaming_mode"] is False
        assert elements[0]["content"] == "**状态：失败**"
        assert elements[1]["content"] == "回答因服务重启中断，请重新发送问题。"
