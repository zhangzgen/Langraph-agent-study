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

from langraph_agent.config import config
from langraph_agent.graph import stream_answer


LOGGER = logging.getLogger(__name__)
CARD_MARKDOWN_ELEMENT_ID = "answer"


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

    def update_card_text(self, card_id: str, text: str, sequence: int) -> None:
        """更新流式卡片内的 markdown 回答内容。

        Description:
            通过 CardKit 流式文本接口以全量覆盖方式刷新回答正文。
        Args:
            card_id (str): 待更新的卡片实体 ID。
            text (str): 截至当前时刻的完整回答正文。
            sequence (int): 当前卡片严格递增的更新序号。
        Returns:
            None: 接口调用成功即完成一次内容刷新。
        """
        self._authorized_request(
            "PUT",
            (
                f"/open-apis/cardkit/v1/cards/{card_id}/elements/"
                f"{CARD_MARKDOWN_ELEMENT_ID}/content"
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
    ) -> None:
        """初始化一张卡片的流式写入器。

        Description:
            记录卡片更新序号、最近更新时间与最新正文，用于合并过于频繁的 token 更新。
        Args:
            client (FeishuOpenAPIClient): 调用 CardKit API 的客户端。
            card_id (str): 当前回答使用的卡片实体 ID。
            update_interval_ms (int): 两次网络更新之间的最小毫秒间隔。
        Returns:
            None: 该方法仅初始化流式写入状态。
        """
        self._client = client
        self._card_id = card_id
        self._interval_seconds = max(update_interval_ms, 0) / 1000
        self._sequence = 0
        self._latest_text = ""
        self._sent_text = ""
        self._last_update_at = 0.0

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
        self._sequence += 1
        self._client.update_card_text(self._card_id, text or " ", self._sequence)
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
        self._sequence += 1
        self._client.finish_streaming_card(self._card_id, self._sequence)


class FeishuBotService:
    """负责接收飞书消息事件并异步生成 Agent 卡片回答。"""

    def __init__(
        self,
        client: FeishuOpenAPIClient,
        answer_streamer: Callable[
            [str, str, Callable[[str], None], str | None], AIMessage
        ] = stream_answer,
        worker_count: int = config.FEISHU_WORKER_COUNT,
        update_interval_ms: int = config.FEISHU_CARD_UPDATE_INTERVAL_MS,
    ) -> None:
        """初始化机器人业务服务。

        Description:
            配置飞书客户端、LangGraph 流式入口以及后台执行池，并创建事件去重与
            按会话串行处理所需的内存状态。
        Args:
            client (FeishuOpenAPIClient): 飞书 OpenAPI 请求客户端。
            answer_streamer (Callable): 以回调方式输出回答的 LangGraph 执行函数。
            worker_count (int): 同时可处理的不同会话任务数量。
            update_interval_ms (int): 卡片正文网络更新最小间隔毫秒数。
        Returns:
            None: 该方法仅初始化服务状态。
        """
        self._client = client
        self._answer_streamer = answer_streamer
        self._update_interval_ms = update_interval_ms
        self._executor = ThreadPoolExecutor(max_workers=max(worker_count, 1))
        self._seen_message_ids: set[str] = set()
        self._seen_order: deque[str] = deque()
        self._chat_locks: dict[str, threading.Lock] = {}
        self._state_lock = threading.Lock()

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
