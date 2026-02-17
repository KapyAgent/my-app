import asyncio
import os

import logfire
import aio_pika
import json
import datetime
import anyio
from typing import Any, TYPE_CHECKING
from k.config import Config
from k.agent.memory.folder import FolderMemoryStore
from k.starters.telegram.runner import _run_agent_for_chat_batch
from k.starters.telegram.compact import (
    dispatch_groups_for_batch,
    _expand_chat_id_watchlist,
)
from k.starters.telegram.api import TelegramBotApi, TelegramBotApiError
from k.starters.telegram.tz import _DEFAULT_TIMEZONE, _parse_timezone

if TYPE_CHECKING:
    from pydantic_ai.models import Model

def _mq_to_update(mq_msg: dict[str, Any]) -> dict[str, Any]:
    """Map custom MQ format to standard Telegram Update format."""
    dt_str = mq_msg.get("date")
    try:
        # Handle ISO format with potential Z or timezone offset
        dt = datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        ts = int(dt.timestamp())
    except Exception:
        ts = int(datetime.datetime.now().timestamp())

    # Build a fake Telegram Update object
    msg = {
        "message_id": mq_msg.get("message_id"),
        "date": ts,
        "chat": {"id": mq_msg.get("chat_id"), "type": "supergroup"},
        "from": {
            "id": mq_msg.get("sender_id"),
            "username": mq_msg.get("sender_username"),
            "first_name": mq_msg.get("sender_fullname"),
            "is_bot": mq_msg.get("is_bot", False),
        },
        "text": mq_msg.get("text"),
        "caption": mq_msg.get("caption"),
    }
    
    if mq_msg.get("is_reply") and mq_msg.get("reply_to"):
        reply = mq_msg["reply_to"]
        msg["reply_to_message"] = {
            "message_id": reply.get("message_id"),
            "from": {
                "id": reply.get("sender_id"),
                "username": reply.get("sender_username"),
                "first_name": reply.get("sender_fullname"),
            },
            "text": reply.get("text"),
        }

    # Use message_id as a surrogate for update_id
    return {"update_id": mq_msg.get("message_id"), "message": msg}

async def run_amqp_forever(
    *,
    config: Config,
    model: Any,
    token: str | None,
    amqp_url: str,
    queue_name: str,
    keyword: str,
    chat_ids: set[int] | None,
    tz: datetime.tzinfo,
):
    mem_store = FolderMemoryStore(root=config.fs_base / "memories")
    append_lock = anyio.Lock()

    bot_user_id = None
    bot_username = None
    if token:
        api = TelegramBotApi(token=token)
        try:
            me = await api.get_me()
            bot_user_id = me.get("id")
            bot_username = me.get("username")
        except TelegramBotApiError as e:
            print(f"[yellow]Telegram getMe failed[/yellow]: {e}")

    print(
        "\n".join(
            [
                "Telegram AMQP listener running.",
                f"- model: {model}",
                f"- amqp_url: {amqp_url}",
                f"- queue_name: {queue_name}",
                f"- keyword: {keyword!r}",
                f"- chat_ids: {sorted(chat_ids) if chat_ids is not None else None}",
                f"- timezone: {tz}",
                f"- bot_user_id: {bot_user_id}",
                f"- bot_username: {bot_username}",
            ]
        )
    )

    connection = await aio_pika.connect_robust(amqp_url)
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=10)
        
        # Ensure we have our own queue to avoid missing messages due to other consumers
        # and bind it to the chats we care about.
        queue = await channel.declare_queue(exclusive=True)
        if chat_ids:
            for cid in chat_ids:
                # Standard routing key format for the userbot-listener
                routing_key = f"chat:{cid}"
                await queue.bind("telegram.messages", routing_key=routing_key)
                print(f"Bound to chat: {cid}")
        else:
            # Fallback to the provided queue name if no chat_ids specified
            queue = await channel.get_queue(queue_name)
        
        pending_updates = []
        
        async with anyio.create_task_group() as tg:
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        try:
                            body = json.loads(message.body.decode())
                            # Convert to standard format
                            update = _mq_to_update(body) if "chat_id" in body else body
                            pending_updates.append(update)
                        except Exception as e:
                            print(f"Failed to parse message: {e}")
                            continue
                        
                        # Check triggers for each keyword in the list (split by |)
                        keywords = [k.strip() for k in keyword.split("|") if k.strip()]
                        
                        # We use a custom trigger check to support multiple keywords
                        def get_triggered_keyword(updates):
                            for k in keywords:
                                if dispatch_groups_for_batch(
                                    updates,
                                    keyword=k,
                                    chat_ids=chat_ids,
                                    bot_user_id=bot_user_id,
                                    bot_username=bot_username,
                                ):
                                    return k
                            return None

                        triggered_keyword = get_triggered_keyword(pending_updates)
                        
                        grouped = None
                        if triggered_keyword:
                            grouped = dispatch_groups_for_batch(
                                pending_updates,
                                keyword=triggered_keyword,
                                chat_ids=chat_ids,
                                bot_user_id=bot_user_id,
                                bot_username=bot_username,
                            )
                        
                        if grouped:
                            print(f"[green]AMQP trigger[/green] pending={len(pending_updates)} groups={len(grouped)}")
                            for cid, updates_for_chat in grouped.items():
                                tg.start_soon(
                                    _run_agent_for_chat_batch,
                                    cid,
                                    list(updates_for_chat),
                                    model,
                                    config,
                                    mem_store,
                                    append_lock,
                                    tz,
                                )
                            pending_updates = []
                        else:
                            if len(pending_updates) > 100:
                                pending_updates.pop(0)

async def run(
    *,
    token: str | None = None,
    amqp_url: str | None = None,
    queue_name: str = "telegram.messages.raw",
    keyword: str,
    model: Any,
    chat_id: str = "",
    timezone: str = _DEFAULT_TIMEZONE,
) -> None:
    if amqp_url is None:
        amqp_url = os.environ.get("AMQP_URL")
    if amqp_url is None:
        raise ValueError("amqp_url is required (pass it directly or set AMQP_URL env var)")
    logfire.configure()
    logfire.instrument_pydantic_ai()


    config = Config()
    try:
        tz = _parse_timezone(str(timezone))
    except ValueError as e:
        raise ValueError(f"Invalid timezone: {e}") from e

    parsed_chat_ids: set[int] | None
    raw_chat_ids = str(chat_id).strip()
    if not raw_chat_ids:
        parsed_chat_ids = None
    else:
        import re
        parts = [p for p in re.split(r"[,\s]+", raw_chat_ids) if p]
        try:
            parsed_chat_ids = {int(p) for p in parts}
        except ValueError as e:
            raise ValueError(f"Invalid chat_id entry in: {raw_chat_ids!r}") from e
        parsed_chat_ids = _expand_chat_id_watchlist(parsed_chat_ids)

    await run_amqp_forever(
        config=config,
        model=model,
        token=token,
        amqp_url=amqp_url,
        queue_name=queue_name,
        keyword=keyword,
        chat_ids=parsed_chat_ids,
        tz=tz,
    )
