from __future__ import annotations

import json

import httpx
from langchain_core.messages import AIMessage

from langraph_agent.feishu_bot import CardStreamWriter, FeishuBotService, FeishuOpenAPIClient


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
        client.finish_streaming_card(card_id, 2)
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
    assert card["body"]["elements"][0]["element_id"] == "answer"
    send_body = requests[2][2]
    assert send_body["msg_type"] == "interactive"
    assert json.loads(send_body["content"])["data"]["card_id"] == "card-1"
    assert requests[3][2]["content"] == "hello"
    assert requests[4][2]["sequence"] == 2


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

    def update_card_text(self, card_id: str, text: str, sequence: int) -> None:
        """记录卡片正文更新动作。

        Description:
            保存模拟流式写入时的文本与序号。
        Args:
            card_id (str): 目标卡片 ID。
            text (str): 完整回答正文。
            sequence (int): 更新序号。
        Returns:
            None: 该方法只记录调用。
        """
        self.calls.append(("update", card_id, text, sequence))

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
    assert ("update", "card-1", "你", 1) in client.calls
    assert ("update", "card-1", "你好", 2) in client.calls
    assert ("finish", "card-1", 3) in client.calls
