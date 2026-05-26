from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from langchain_core.messages import AIMessage
from langgraph.types import Command

from langraph_agent.config import config
from langraph_agent.feishu_approvals import (
    CardContentBlock,
    CardSession,
    DisplayToolCall,
    FeishuApprovalStore,
)
from langraph_agent.graph import ChannelRunOutcome, stream_answer_until_interrupt


LOGGER = logging.getLogger(__name__)
CARD_MARKDOWN_ELEMENT_ID = "answer"
CARD_STATUS_ELEMENT_ID = "status"


class FeishuAPIError(RuntimeError):
    """表示飞书开放平台接口返回失败结果。"""


class FeishuOpenAPIClient:
    """封装飞书消息与 CardKit OpenAPI 的轻量客户端。"""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        base_url: str = config.FEISHU_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """初始化飞书 OpenAPI 客户端。

        Description:
            保存应用身份信息，创建复用连接的 HTTP 客户端，并初始化租户凭证缓存。
        Args:
            app_id (str): 飞书应用 App ID。
            app_secret (str): 飞书应用 App Secret。
            base_url (str): 飞书开放平台 API 根地址。
            transport (httpx.BaseTransport | None): 测试或定制网络请求使用的传输器。
        Returns:
            None: 该方法仅初始化客户端状态。
        """
        self._app_id = app_id
        self._app_secret = app_secret
        self._client = httpx.Client(
            base_url=base_url,
            timeout=20.0,
            transport=transport,
        )
        self._tenant_token = ""
        self._tenant_token_expires_at = 0.0
        self._token_lock = threading.Lock()

    def close(self) -> None:
        """关闭底层 HTTP 连接池。

        Description:
            释放 httpx 客户端持有的网络连接资源。
        Args:
            无。
        Returns:
            None: 该方法没有业务返回值。
        """
        self._client.close()

    def create_streaming_card(self) -> str:
        """创建用于逐步展示回答的 CardKit 卡片实体。

        Description:
            创建 Card JSON 2.0 卡片并打开 streaming_mode，卡片包含可持续更新的
            markdown 正文元素。
        Args:
            无。
        Returns:
            str: 飞书返回的卡片实体 ID。
        """
        card = {
            "schema": "2.0",
            "config": {
                "update_multi": True,
                "streaming_mode": True,
                "summary": {"content": "正在生成回答"},
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "element_id": CARD_STATUS_ELEMENT_ID,
                        "content": "**状态：生成中**",
                    },
                    {
                        "tag": "markdown",
                        "element_id": CARD_MARKDOWN_ELEMENT_ID,
                        "content": "正在生成回答...",
                    }
                ]
            },
        }
        data = self._authorized_request(
            "POST",
            "/open-apis/cardkit/v1/cards",
            {"type": "card_json", "data": json.dumps(card, ensure_ascii=False)},
        )
        card_id = data.get("card_id")
        if not isinstance(card_id, str) or not card_id:
            raise FeishuAPIError("创建飞书卡片成功，但响应缺少 card_id。")
        return card_id

    def replace_card(self, card_id: str, card: dict[str, Any], sequence: int) -> None:
        """使用完整 Card JSON 覆盖卡片内容。

        Description:
            在状态切换或呈现审批按钮时更新整张 CardKit 卡片，可同时控制
            `streaming_mode` 以满足交互卡片触发前必须结束流式状态的约束。
        Args:
            card_id (str): 待覆盖的卡片实体 ID。
            card (dict[str, Any]): 完整 Card JSON 2.0 数据。
            sequence (int): 当前卡片严格递增的更新序号。
        Returns:
            None: 接口成功即完成一次卡片覆盖。
        """
        self._authorized_request(
            "PUT",
            f"/open-apis/cardkit/v1/cards/{card_id}",
            {
                "card": {
                    "type": "card_json",
                    "data": json.dumps(card, ensure_ascii=False),
                },
                "sequence": sequence,
                "uuid": uuid.uuid4().hex,
            },
        )

    def send_card(self, chat_id: str, card_id: str) -> None:
        """向飞书单聊会话发送卡片实体消息。

        Description:
            使用 chat_id 作为接收者，把已创建的 CardKit 卡片实体发送到当前会话。
        Args:
            chat_id (str): 用户与机器人的飞书单聊会话 ID。
            card_id (str): 已创建的 CardKit 卡片实体 ID。
        Returns:
            None: 接口调用成功即完成发送。
        """
        content = {"type": "card", "data": {"card_id": card_id}}
        self._authorized_request(
            "POST",
            "/open-apis/im/v1/messages?receive_id_type=chat_id",
            {
                "receive_id": chat_id,
                "msg_type": "interactive",
                "content": json.dumps(content, ensure_ascii=False),
                "uuid": uuid.uuid4().hex,
            },
        )

    def send_text(self, chat_id: str, text: str) -> None:
        """向飞书会话发送普通文本提示。

        Description:
            在无法建立卡片回答时向用户发送可见的降级错误提示或输入提示。
        Args:
            chat_id (str): 用户与机器人的飞书单聊会话 ID。
            text (str): 要展示给用户的文本正文。
        Returns:
            None: 接口调用成功即完成发送。
        """
        self._authorized_request(
            "POST",
            "/open-apis/im/v1/messages?receive_id_type=chat_id",
            {
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
                "uuid": uuid.uuid4().hex,
            },
        )

    def update_card_text(
        self,
        card_id: str,
        text: str,
        sequence: int,
        element_id: str = CARD_MARKDOWN_ELEMENT_ID,
    ) -> None:
        """更新流式卡片内的 markdown 回答内容。

        Description:
            通过 CardKit 流式文本接口以全量覆盖方式刷新回答正文。
        Args:
            card_id (str): 待更新的卡片实体 ID。
            text (str): 截至当前时刻的完整回答正文。
            sequence (int): 当前卡片严格递增的更新序号。
            element_id (str): 当前需要流式刷新的 markdown 内容块 ID。
        Returns:
            None: 接口调用成功即完成一次内容刷新。
        """
        self._authorized_request(
            "PUT",
            (
                f"/open-apis/cardkit/v1/cards/{card_id}/elements/"
                f"{element_id}/content"
            ),
            {
                "content": text,
                "sequence": sequence,
                "uuid": uuid.uuid4().hex,
            },
        )

    def finish_streaming_card(self, card_id: str, sequence: int) -> None:
        """关闭卡片的流式显示状态。

        Description:
            在回答输出完成后更新卡片配置，关闭光标等流式展示效果。
        Args:
            card_id (str): 待结束流式状态的卡片实体 ID。
            sequence (int): 当前卡片严格递增的更新序号。
        Returns:
            None: 接口调用成功即标记卡片回答完成。
        """
        settings = {
            "config": {
                "streaming_mode": False,
                "summary": {"content": "回答已完成"},
            }
        }
        self._authorized_request(
            "PATCH",
            f"/open-apis/cardkit/v1/cards/{card_id}/settings",
            {
                "settings": json.dumps(settings, ensure_ascii=False),
                "sequence": sequence,
                "uuid": uuid.uuid4().hex,
            },
        )

    def _get_tenant_access_token(self) -> str:
        """获取并缓存应用租户访问凭证。

        Description:
            调用飞书内部应用凭证接口，提前一分钟刷新即将过期的 token，
            避免每次卡片更新都重新认证。
        Args:
            无。
        Returns:
            str: 可用于调用应用 OpenAPI 的 tenant_access_token。
        """
        with self._token_lock:
            now = time.monotonic()
            if self._tenant_token and now < self._tenant_token_expires_at:
                return self._tenant_token

            response = self._client.post(
                "/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": self._app_id, "app_secret": self._app_secret},
            )
            payload = self._parse_response(response)
            token = payload.get("tenant_access_token")
            if not isinstance(token, str) or not token:
                raise FeishuAPIError("飞书认证响应缺少 tenant_access_token。")
            expire = payload.get("expire")
            expires_in = expire if isinstance(expire, int) else 7200
            self._tenant_token = token
            self._tenant_token_expires_at = now + max(expires_in - 60, 1)
            return token

    def _authorized_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """执行带租户身份的飞书 OpenAPI 请求。

        Description:
            为 API 请求添加 Bearer token，并统一校验 HTTP 状态及飞书业务响应码。
        Args:
            method (str): HTTP 方法名称。
            path (str): 飞书 OpenAPI 相对请求路径。
            payload (dict[str, Any]): 要以 JSON 形式发送的请求体。
        Returns:
            dict[str, Any]: 飞书业务响应中的 data 对象，缺省时为空字典。
        """
        response = self._client.request(
            method,
            path,
            headers={"Authorization": f"Bearer {self._get_tenant_access_token()}"},
            json=payload,
        )
        response_payload = self._parse_response(response)
        data = response_payload.get("data")
        return data if isinstance(data, dict) else {}

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        """解析并校验飞书开放平台响应。

        Description:
            确保 HTTP 请求与飞书业务 code 均表示成功，否则抛出可记录的异常。
        Args:
            response (httpx.Response): 飞书接口返回的 HTTP 响应对象。
        Returns:
            dict[str, Any]: 已确认业务成功的 JSON 响应体。
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise FeishuAPIError(
                f"飞书接口返回非 JSON 响应，HTTP {response.status_code}。"
            ) from exc
        if response.status_code >= 400 or payload.get("code") != 0:
            raise FeishuAPIError(
                f"飞书接口调用失败，HTTP {response.status_code}: "
                f"{payload.get('code')} {payload.get('msg', '')}"
            )
        return payload


class CardStreamWriter:
    """对 CardKit 全量文本更新进行限频与完成态收口。"""

    def __init__(
        self,
        client: FeishuOpenAPIClient,
        card_id: str,
        update_interval_ms: int,
        element_id: str = CARD_MARKDOWN_ELEMENT_ID,
        sequence_provider: Callable[[], int] | None = None,
    ) -> None:
        """初始化一张卡片的流式写入器。

        Description:
            记录卡片更新序号、最近更新时间与最新正文，用于合并过于频繁的 token 更新。
        Args:
            client (FeishuOpenAPIClient): 调用 CardKit API 的客户端。
            card_id (str): 当前回答使用的卡片实体 ID。
            update_interval_ms (int): 两次网络更新之间的最小毫秒间隔。
            element_id (str): 当前写入器需要更新的 markdown 内容块 ID。
            sequence_provider (Callable[[], int] | None): 可选的持久化序号生成器。
        Returns:
            None: 该方法仅初始化流式写入状态。
        """
        self._client = client
        self._card_id = card_id
        self._element_id = element_id
        self._interval_seconds = max(update_interval_ms, 0) / 1000
        self._sequence = 0
        self._latest_text = ""
        self._sent_text = ""
        self._last_update_at = 0.0
        self._sequence_provider = sequence_provider

    def write(self, text: str, force: bool = False) -> None:
        """写入当前累计回答正文。

        Description:
            保存最新完整文本，并按节流间隔触发 CardKit 内容更新；force 为 True
            时立即发送最后一次尚未刷新的正文。
        Args:
            text (str): 模型截至当前已生成的完整回答正文。
            force (bool): 是否绕过更新间隔立即刷新。
        Returns:
            None: 该方法通过远端卡片展示结果。
        """
        self._latest_text = text
        now = time.monotonic()
        if not force and now - self._last_update_at < self._interval_seconds:
            return
        sequence = self._next_sequence()
        self._client.update_card_text(
            self._card_id,
            text or " ",
            sequence,
            element_id=self._element_id,
        )
        self._sent_text = text
        self._last_update_at = now

    def finish(self) -> None:
        """刷新最终正文并结束卡片流式状态。

        Description:
            提交尚未发送的最终正文，然后递增 sequence 并关闭 streaming_mode。
        Args:
            无。
        Returns:
            None: 该方法完成卡片最终呈现。
        """
        final_text = self._latest_text or "未生成可展示的回答。"
        if final_text != self._sent_text:
            self.write(final_text, force=True)
        self._client.finish_streaming_card(self._card_id, self._next_sequence())

    def _next_sequence(self) -> int:
        """取得下一次 CardKit 更新的序号。

        Description:
            优先使用持久化序号提供器，使流式文本与整卡状态更新共享单调序列；
            兼容旧调用时继续使用写入器本地计数。
        Args:
            无。
        Returns:
            int: 可用于当前卡片下一次更新的递增序号。
        """
        if self._sequence_provider is not None:
            return self._sequence_provider()
        self._sequence += 1
        return self._sequence


def build_interactive_card(
    session: CardSession,
    tool_calls: list[DisplayToolCall],
    content_blocks: list[CardContentBlock],
) -> dict[str, Any]:
    """构建回答和逐项工具审批使用的 CardKit 卡片。

    Description:
        根据持久化时间线依次输出模型正文和工具调用块；工具块使用分隔线包围，
        当前待审工具在原位置插入横向排列的批准与拒绝按钮。
    Args:
        session (CardSession): 当前卡片会话状态与已生成正文。
        tool_calls (list[DisplayToolCall]): 已观察到的工具调用展示条目。
        content_blocks (list[CardContentBlock]): 按发生顺序持久化的正文与工具内容块。
    Returns:
        dict[str, Any]: 可通过整卡覆盖接口发送的 Card JSON 2.0。
    """
    status_labels = {
        "generating": "生成中",
        "pending_approval": "待审批",
        "executing": "执行中",
        "completed": "完成",
        "failed": "失败",
    }
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "element_id": CARD_STATUS_ELEMENT_ID,
            "content": f"**状态：{status_labels[session['status']]}**",
        },
    ]
    if not content_blocks:
        elements.append(
            {
                "tag": "markdown",
                "element_id": CARD_MARKDOWN_ELEMENT_ID,
                "content": session["answer_text"] or "正在生成回答...",
            }
        )
    tool_status_labels = {
        "pending": "待审批",
        "auto_approved": "自动执行",
        "approved": "已批准",
        "rejected": "已拒绝",
    }
    tools_by_id = {item["tool_call_id"]: item for item in tool_calls}
    pending = next((item for item in tool_calls if item["status"] == "pending"), None)
    for block in content_blocks:
        if block["block_type"] == "text":
            elements.append(
                {
                    "tag": "markdown",
                    "element_id": f"answer_{block['block_id']}",
                    "content": block["content"] or " ",
                }
            )
            continue
        tool_call = tools_by_id.get(str(block["tool_call_id"] or ""))
        if tool_call is None:
            continue
        if elements[-1]["tag"] != "hr":
            elements.append({"tag": "hr"})
        description = (
            f"**工具调用：`{tool_call['tool_name']}`** "
            f"({tool_status_labels[tool_call['status']]})"
        )
        if tool_call["display_content"]:
            description += f"\n{tool_call['display_content']}"
        elements.append({"tag": "markdown", "content": description})
        if (
            session["status"] == "pending_approval"
            and pending is not None
            and tool_call["tool_call_id"] == pending["tool_call_id"]
        ):
            elements.extend(
                [
                    {"tag": "markdown", "content": "请审核当前工具调用："},
                    {
                        "tag": "column_set",
                        "horizontal_spacing": "8px",
                        "columns": [
                            {
                                "tag": "column",
                                "width": "weighted",
                                "weight": 1,
                                "elements": [
                                    {
                                        "tag": "button",
                                        "text": {"tag": "plain_text", "content": "批准"},
                                        "type": "primary",
                                        "width": "fill",
                                        "behaviors": [
                                            {
                                                "type": "callback",
                                                "value": {
                                                    "action": "approve_tool",
                                                    "card_id": session["card_id"],
                                                    "tool_call_id": pending["tool_call_id"],
                                                    "action_key": (
                                                        f"{session['card_id']}:"
                                                        f"{pending['tool_call_id']}:approve"
                                                    ),
                                                },
                                            }
                                        ],
                                    }
                                ],
                            },
                            {
                                "tag": "column",
                                "width": "weighted",
                                "weight": 1,
                                "elements": [
                                    {
                                        "tag": "button",
                                        "text": {"tag": "plain_text", "content": "拒绝"},
                                        "type": "danger",
                                        "width": "fill",
                                        "behaviors": [
                                            {
                                                "type": "callback",
                                                "value": {
                                                    "action": "reject_tool",
                                                    "card_id": session["card_id"],
                                                    "tool_call_id": pending["tool_call_id"],
                                                    "action_key": (
                                                        f"{session['card_id']}:"
                                                        f"{pending['tool_call_id']}:reject"
                                                    ),
                                                },
                                            }
                                        ],
                                    }
                                ],
                            },
                        ],
                    },
                ]
            )
        elements.append({"tag": "hr"})
    return {
        "schema": "2.0",
        "config": {
            "update_multi": True,
            "streaming_mode": session["status"] in {"generating", "executing"},
            "summary": {"content": f"回答{status_labels[session['status']]}"},
        },
        "body": {"elements": elements},
    }


class FeishuBotService:
    """负责接收飞书消息事件并异步生成 Agent 卡片回答。"""

    def __init__(
        self,
        client: FeishuOpenAPIClient,
        answer_streamer: Callable[
            [str, str, Callable[[str], None], str | None], AIMessage
        ]
        | None = None,
        worker_count: int = config.FEISHU_WORKER_COUNT,
        update_interval_ms: int = config.FEISHU_CARD_UPDATE_INTERVAL_MS,
        agent_runner: Callable[
            [
                str | Command,
                str,
                Callable[[str], None],
                Callable[[list[dict[str, Any]]], None],
                str | None,
            ],
            ChannelRunOutcome,
        ] = stream_answer_until_interrupt,
        approval_store: FeishuApprovalStore | None = None,
    ) -> None:
        """初始化机器人业务服务。

        Description:
            配置飞书客户端、LangGraph 流式入口以及后台执行池，并创建事件去重与
            按会话串行处理所需的内存状态。
        Args:
            client (FeishuOpenAPIClient): 飞书 OpenAPI 请求客户端。
            answer_streamer (Callable | None): 兼容旧流式回答测试或集成的执行函数。
            worker_count (int): 同时可处理的不同会话任务数量。
            update_interval_ms (int): 卡片正文网络更新最小间隔毫秒数。
            agent_runner (Callable): 支持 interrupt 暂停和恢复的 LangGraph 渠道执行函数。
            approval_store (FeishuApprovalStore | None): 飞书审批状态持久化存储。
        Returns:
            None: 该方法仅初始化服务状态。
        """
        self._client = client
        self._answer_streamer = answer_streamer
        self._agent_runner = agent_runner
        self._approval_store = approval_store or (
            None if answer_streamer is not None else FeishuApprovalStore()
        )
        self._update_interval_ms = update_interval_ms
        self._executor = ThreadPoolExecutor(max_workers=max(worker_count, 1))
        self._seen_message_ids: set[str] = set()
        self._seen_order: deque[str] = deque()
        self._chat_locks: dict[str, threading.Lock] = {}
        self._state_lock = threading.Lock()
        if self._approval_store is not None:
            for session in self._approval_store.list_recoverable_sessions():
                self._executor.submit(self._continue_tool_action, session["card_id"])

    def on_message(self, data: Any) -> None:
        """消费飞书接收消息事件并快速投递后台任务。

        Description:
            只响应用户在机器人单聊发送的文本消息；事件回调不等待模型执行，
            从而及时完成长连接事件确认并避免飞书重试。
        Args:
            data (Any): lark-oapi 转换后的 P2ImMessageReceiveV1 事件数据。
        Returns:
            None: 实际回答由后台线程发送至飞书会话。
        """
        event = getattr(data, "event", None)
        message = getattr(event, "message", None)
        sender = getattr(event, "sender", None)
        message_id = str(getattr(message, "message_id", "") or "")
        chat_id = str(getattr(message, "chat_id", "") or "")
        if (
            not message_id
            or not chat_id
            or getattr(message, "chat_type", "") != "p2p"
            or getattr(sender, "sender_type", "") == "app"
        ):
            return
        if not self._claim_message(message_id):
            return

        if getattr(message, "message_type", "") != "text":
            self._executor.submit(
                self._client.send_text,
                chat_id,
                "当前仅支持文本消息，请输入问题后重试。",
            )
            return
        try:
            body = json.loads(getattr(message, "content", "") or "{}")
        except json.JSONDecodeError:
            body = {}
        question = str(body.get("text") or "").strip()
        if question:
            self._executor.submit(self._answer_message, chat_id, question)

    def on_card_action(self, data: Any) -> Any:
        """消费 CardKit 审批按钮触发事件。

        Description:
            从 `p2.card.action.trigger` 提取批准或拒绝动作，在返回长连接确认前以
            短事务保存决定，再由后台线程刷新卡片或恢复 LangGraph。
        Args:
            data (Any): lark-oapi 转换后的卡片 action 触发事件。
        Returns:
            Any: lark-oapi 所需的空卡片动作响应对象。
        """
        from lark_oapi.event.callback.model.p2_card_action_trigger import (
            P2CardActionTriggerResponse,
        )

        event = getattr(data, "event", None)
        action = getattr(event, "action", None)
        value = getattr(action, "value", None)
        if isinstance(value, dict) and value.get("action") in {
            "approve_tool",
            "reject_tool",
        }:
            try:
                card_id = self._persist_tool_action(value)
                if card_id is not None:
                    self._executor.submit(self._continue_tool_action, card_id)
            except Exception:
                LOGGER.exception("飞书卡片动作持久化失败，value=%s", value)
        return P2CardActionTriggerResponse({})

    def close(self) -> None:
        """关闭机器人业务服务持有的资源。

        Description:
            等待后台回答任务结束并关闭复用的飞书 HTTP 客户端。
        Args:
            无。
        Returns:
            None: 该方法仅执行资源释放。
        """
        self._executor.shutdown(wait=True)
        self._client.close()

    def _claim_message(self, message_id: str) -> bool:
        """记录已接收消息并判断是否首次处理。

        Description:
            在内存中缓存最近一千个消息 ID，过滤长连接重投造成的重复回答。
        Args:
            message_id (str): 飞书消息唯一 ID。
        Returns:
            bool: 首次看到消息时返回 True，重复消息返回 False。
        """
        with self._state_lock:
            if message_id in self._seen_message_ids:
                return False
            self._seen_message_ids.add(message_id)
            self._seen_order.append(message_id)
            if len(self._seen_order) > 1000:
                oldest = self._seen_order.popleft()
                self._seen_message_ids.discard(oldest)
            return True

    def _answer_message(self, chat_id: str, question: str) -> None:
        """为一条用户消息生成并发送流式卡片回答。

        Description:
            针对同一飞书单聊串行执行 LangGraph，将 chat_id 固定映射为 checkpoint
            thread_id，并在异常时向用户展示可理解的失败提示。
        Args:
            chat_id (str): 当前飞书单聊会话 ID。
            question (str): 用户发送的文本问题。
        Returns:
            None: 回答结果直接写入飞书消息卡片。
        """
        with self._state_lock:
            chat_lock = self._chat_locks.setdefault(chat_id, threading.Lock())
        with chat_lock:
            if self._answer_streamer is None:
                self._start_interactive_answer(chat_id, question)
                return
            writer: CardStreamWriter | None = None
            try:
                card_id = self._client.create_streaming_card()
                self._client.send_card(chat_id, card_id)
                writer = CardStreamWriter(
                    self._client,
                    card_id,
                    update_interval_ms=self._update_interval_ms,
                )
                self._answer_streamer(
                    question,
                    f"feishu:{chat_id}",
                    writer.write,
                    None,
                )
                writer.finish()
            except Exception:
                LOGGER.exception("飞书回答生成或发送失败，chat_id=%s", chat_id)
                failure_text = "回答生成失败，请稍后重试。"
                if writer is not None:
                    try:
                        writer.write(failure_text, force=True)
                        writer.finish()
                    except Exception:
                        LOGGER.exception("飞书失败提示卡片更新失败，chat_id=%s", chat_id)
                else:
                    try:
                        self._client.send_text(chat_id, failure_text)
                    except Exception:
                        LOGGER.exception("飞书失败提示消息发送失败，chat_id=%s", chat_id)

    def _start_interactive_answer(self, chat_id: str, question: str) -> None:
        """创建可暂停审批的单卡片回答。

        Description:
            拦截同一聊天中仍在处理的轮次，随后建立卡片与 checkpoint thread 映射，
            并运行首个 Agent 片段直到完成或进入待审批状态。
        Args:
            chat_id (str): 当前飞书单聊会话 ID。
            question (str): 用户发送的文本问题。
        Returns:
            None: 卡片和持久化状态由该流程直接更新。
        """
        store = self._require_approval_store()
        card_id = ""
        try:
            active = store.find_active_session(chat_id)
            if active is not None:
                self._client.send_text(chat_id, "当前回答仍在处理中，请先完成卡片中的审批。")
                return
            card_id = self._client.create_streaming_card()
            session = store.create_session(card_id, chat_id, f"feishu:{chat_id}")
            self._client.send_card(chat_id, card_id)
            self._run_card_step(session, question)
        except Exception:
            LOGGER.exception("飞书交互卡片回答失败，chat_id=%s", chat_id)
            if card_id:
                try:
                    store.set_status(card_id, "failed")
                    store.update_answer(card_id, "回答生成失败，请稍后重试。")
                    self._replace_persisted_card(card_id)
                    return
                except Exception:
                    LOGGER.exception("飞书交互失败状态更新失败，card_id=%s", card_id)
            self._client.send_text(chat_id, "回答生成失败，请稍后重试。")

    def _run_card_step(self, session: CardSession, next_input: str | Command) -> None:
        """执行一段卡片关联的 Agent 流程。

        Description:
            将流式正文写回同一张卡片，记录工具名称；图完成时呈现完成态，
            工具审批中断时关闭流式模式并呈现第一项可审核按钮。
        Args:
            session (CardSession): 正在执行或恢复的持久化卡片会话。
            next_input (str | Command): 用户问题或审批聚合后的恢复命令。
        Returns:
            None: 执行结果以卡片与数据库状态体现。
        """
        store = self._require_approval_store()
        writer: CardStreamWriter | None = None
        active_text_block_id: int | None = None
        text_prefix = ""
        latest_step_text = ""

        def write_text(text: str) -> None:
            """持久化并刷新模型可见正文。

            Description:
                将当前模型输出段落追加或更新到内容时间线；工具调用出现后，后续
                输出会进入新的正文块，从而保留卡片上方已经生成的内容。
            Args:
                text (str): 模型当前执行片段累计产生的正文。
            Returns:
                None: 内容立即或按限频规则更新到远端卡片。
            """
            nonlocal active_text_block_id, latest_step_text, writer
            latest_step_text = text
            segment_text = (
                text[len(text_prefix) :]
                if text_prefix and text.startswith(text_prefix)
                else text
            )
            if not segment_text:
                return
            store.update_answer(session["card_id"], text)
            if active_text_block_id is None:
                active_text_block_id = store.append_text_block(
                    session["card_id"],
                    segment_text,
                )
                self._replace_persisted_card(session["card_id"])
                writer = CardStreamWriter(
                    self._client,
                    session["card_id"],
                    update_interval_ms=self._update_interval_ms,
                    element_id=f"answer_{active_text_block_id}",
                    sequence_provider=lambda: store.next_sequence(session["card_id"]),
                )
                return
            store.update_text_block(active_text_block_id, segment_text)
            if writer is not None:
                writer.write(segment_text)

        def show_tools(tool_calls: list[dict[str, Any]]) -> None:
            """持久化并显示自动执行工具阶段。

            Description:
                在当前正文之后追加工具调用块，并将后续模型输出路由到新的正文块；
                没有待审批项时立即把卡片切换为执行中。
            Args:
                tool_calls (list[dict[str, Any]]): 图分类节点产生的工具调用概要。
            Returns:
                None: 工具列表和卡片阶段通过持久化及远端更新保存。
            """
            nonlocal active_text_block_id, text_prefix, writer
            store.record_tool_calls(session["card_id"], tool_calls)
            text_prefix = latest_step_text
            active_text_block_id = None
            writer = None
            if not any(call.get("approval_required") for call in tool_calls):
                store.set_status(session["card_id"], "executing")
                self._replace_persisted_card(session["card_id"])

        outcome = self._agent_runner(
            next_input,
            session["thread_id"],
            write_text,
            show_tools,
            None,
        )
        if outcome.interrupt_request is not None:
            request = outcome.interrupt_request
            if request.get("type") != "tool_approval":
                raise RuntimeError("飞书渠道暂不支持该类型的交互中断。")
            store.record_tool_calls(
                session["card_id"],
                [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "approval_required": True,
                    }
                    for item in request.get("tool_calls", [])
                ],
            )
            store.set_status(session["card_id"], "pending_approval")
            self._replace_persisted_card(session["card_id"])
            return
        store.set_status(session["card_id"], "completed")
        self._replace_persisted_card(session["card_id"])

    def _handle_tool_action(self, value: dict[str, Any]) -> None:
        """处理一项卡片工具批准或拒绝动作。

        Description:
            同步持久化一项按钮决定，并在本线程继续执行对应的卡片刷新或图恢复；
            该入口主要用于后台调用及测试，长连接回调会先持久化再异步继续。
        Args:
            value (dict[str, Any]): 按钮 value 中携带的动作及工具映射信息。
        Returns:
            None: 审批决定与恢复执行结果写回原卡片。
        """
        card_id = self._persist_tool_action(value)
        if card_id is not None:
            self._continue_tool_action(card_id)

    def _persist_tool_action(self, value: dict[str, Any]) -> str | None:
        """在卡片 action 响应前持久化工具决定。

        Description:
            校验按钮映射并原子写入当前工具结论，使长连接确认返回后即使服务退出，
            后续启动扫描也能发现并恢复已完成审批的会话。
        Args:
            value (dict[str, Any]): 按钮 value 中携带的动作及工具映射信息。
        Returns:
            str | None: 需要后台继续处理的 card_id；无效或无需重复处理时返回 None。
        """
        card_id = str(value.get("card_id") or "")
        tool_call_id = str(value.get("tool_call_id") or "")
        action_key = str(value.get("action_key") or "")
        if not card_id or not tool_call_id or not action_key:
            return None
        store = self._require_approval_store()
        try:
            session = store.get_session(card_id)
        except KeyError:
            return None
        applied = store.decide_tool(
            card_id,
            tool_call_id,
            approved=value.get("action") == "approve_tool",
            action_key=action_key,
        )
        remaining_pending = any(
            item["status"] == "pending" for item in store.list_tool_calls(card_id)
        )
        if applied:
            return card_id
        if (
            session["status"] in {"pending_approval", "executing"}
            and not remaining_pending
        ):
            return card_id
        return None

    def _continue_tool_action(self, card_id: str) -> None:
        """刷新下一项审批或恢复全部已决策的图执行。

        Description:
            在后台按聊天串行处理已持久化的决定；仍存在待审工具时只刷新按钮，
            否则切换为执行中并使用聚合批准 ID 恢复 LangGraph。
        Args:
            card_id (str): 已记录用户决定的 CardKit 卡片实体标识。
        Returns:
            None: 后续状态更新与 Agent 输出继续写入同一卡片。
        """
        store = self._require_approval_store()
        try:
            session = store.get_session(card_id)
        except KeyError:
            return
        with self._state_lock:
            chat_lock = self._chat_locks.setdefault(session["chat_id"], threading.Lock())
        with chat_lock:
            try:
                remaining_pending = any(
                    item["status"] == "pending" for item in store.list_tool_calls(card_id)
                )
                if store.get_session(card_id)["status"] not in {
                    "pending_approval",
                    "executing",
                }:
                    return
                if remaining_pending:
                    self._replace_persisted_card(card_id)
                    return
                store.set_status(card_id, "executing")
                self._replace_persisted_card(card_id)
                refreshed = store.get_session(card_id)
                self._run_card_step(
                    refreshed,
                    Command(resume={"approved_call_ids": store.approved_call_ids(card_id)}),
                )
            except Exception:
                LOGGER.exception("飞书工具审批恢复失败，card_id=%s", card_id)
                try:
                    store.set_status(card_id, "failed")
                    store.update_answer(card_id, "审批处理失败，请稍后重试。")
                    self._replace_persisted_card(card_id)
                except Exception:
                    LOGGER.exception("飞书审批失败状态更新失败，card_id=%s", card_id)

    def _replace_persisted_card(self, card_id: str) -> None:
        """使用数据库中的最新会话状态覆盖远端卡片。

        Description:
            统一渲染状态栏、回答正文、工具列表和当前审批按钮，并使用持久化
            sequence 调用 CardKit 整卡更新接口。
        Args:
            card_id (str): CardKit 卡片实体标识。
        Returns:
            None: 成功后飞书展示即反映最新持久化状态。
        """
        store = self._require_approval_store()
        session = store.get_session(card_id)
        self._client.replace_card(
            card_id,
            build_interactive_card(
                session,
                store.list_tool_calls(card_id),
                store.list_content_blocks(card_id),
            ),
            store.next_sequence(card_id),
        )

    def _require_approval_store(self) -> FeishuApprovalStore:
        """取得交互审批流程所需的持久化存储。

        Description:
            为仅在旧测试兼容路径下省略存储的服务实例提供清晰的失败提示。
        Args:
            无。
        Returns:
            FeishuApprovalStore: 当前服务配置的审批存储对象。
        """
        if self._approval_store is None:
            raise RuntimeError("当前飞书服务未配置交互审批存储。")
        return self._approval_store


def build_event_handler(service: FeishuBotService) -> Any:
    """构建飞书 SDK 长连接事件分发器。

    Description:
        注册 `im.message.receive_v1` 处理函数，使长连接收到用户消息后交给
        FeishuBotService 快速投递。
    Args:
        service (FeishuBotService): 负责实际回复逻辑的机器人服务实例。
    Returns:
        Any: lark-oapi 可交给 WebSocket Client 使用的事件分发器。
    """
    import lark_oapi as lark

    return (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(service.on_message)
        .register_p2_card_action_trigger(service.on_card_action)
        .build()
    )


def run_feishu_bot() -> None:
    """启动飞书长连接机器人服务。

    Description:
        校验环境变量中的应用凭证，初始化 CardKit 客户端和 lark-oapi
        WebSocket 客户端，并持续监听飞书消息事件。
    Args:
        无。
    Returns:
        None: 函数持续运行直至长连接客户端退出或收到中断。
    """
    if not config.FEISHU_APP_ID or not config.FEISHU_APP_SECRET:
        raise RuntimeError("缺少 FEISHU_APP_ID 或 FEISHU_APP_SECRET 环境变量。")

    import lark_oapi as lark

    service = FeishuBotService(
        FeishuOpenAPIClient(config.FEISHU_APP_ID, config.FEISHU_APP_SECRET)
    )
    event_handler = build_event_handler(service)
    ws_client = lark.ws.Client(
        config.FEISHU_APP_ID,
        config.FEISHU_APP_SECRET,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )
    try:
        ws_client.start()
    finally:
        service.close()


def main() -> None:
    """运行飞书机器人的命令行入口。

    Description:
        初始化基础日志输出后启动飞书长连接机器人。
    Args:
        无。
    Returns:
        None: 入口函数在服务退出后结束。
    """
    logging.basicConfig(level=logging.INFO)
    run_feishu_bot()


if __name__ == "__main__":
    main()
