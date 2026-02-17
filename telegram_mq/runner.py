import os

import logfire
import aio_pika
import json
import datetime
import anyio
import anyio.to_thread as to_thread
from functools import partial
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from k.config import Config
from k.agent.memory.folder import FolderMemoryStore
from k.starters.telegram.runner import (
    run_agent_for_chat_batch,
    overlay_dispatch_groups_with_recent,
    filter_dispatch_groups_after_last_trigger,
    update_last_trigger_update_id_by_chat,
)
from k.starters.telegram.compact import (
    dispatch_groups_for_batch,
    _expand_chat_id_watchlist,
    extract_update_id,
    filter_unseen_updates,
)
from k.starters.telegram.api import TelegramBotApi, TelegramBotApiError
from k.starters.telegram.tz import _DEFAULT_TIMEZONE, _parse_timezone
from k.starters.telegram.history import (
    append_updates_jsonl,
    load_last_trigger_update_id_by_chat,
    load_recent_updates_grouped_by_chat_id,
    save_last_trigger_update_id_by_chat,
    trigger_cursor_state_path_for_updates_store,
)

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
    updates_store_path: Optional[Path] = None,
    dispatch_recent_per_chat: int = 0,
    tz: datetime.tzinfo,
):
    if dispatch_recent_per_chat < 0:
        raise ValueError(
            f"dispatch_recent_per_chat must be >= 0; got {dispatch_recent_per_chat}"
        )
    if dispatch_recent_per_chat > 0 and updates_store_path is None:
        raise ValueError(
            "dispatch_recent_per_chat requires updates_store_path to be configured"
        )

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

    last_consumed_update_id: int | None = None
    pending_updates_by_id: dict[int, dict[str, Any]] = {}

    last_trigger_update_id_by_chat: dict[int, int] = {}
    trigger_cursor_state_path: Path | None = None
    if updates_store_path is not None:
        trigger_cursor_state_path = trigger_cursor_state_path_for_updates_store(
            updates_store_path
        )
        try:
            last_trigger_update_id_by_chat = await to_thread.run_sync(
                load_last_trigger_update_id_by_chat,
                trigger_cursor_state_path,
            )
        except (OSError, ValueError) as e:
            print(
                "[yellow]telegram trigger cursor load error[/yellow] "
                + f"path={trigger_cursor_state_path}: {type(e).__name__}: {e}"
            )
            last_trigger_update_id_by_chat = {}

    print(
        "\n".join(
            [
                "Telegram AMQP listener running.",
                f"- model: {model}",
                f"- amqp_url: {amqp_url}",
                f"- queue_name: {queue_name}",
                f"- keyword: {keyword!r}",
                f"- chat_ids: {sorted(chat_ids) if chat_ids is not None else None}",
                f"- updates_store_path: {updates_store_path}",
                f"- trigger_cursor_state_path: {trigger_cursor_state_path}",
                f"- loaded_trigger_cursor_chats: {len(last_trigger_update_id_by_chat)}",
                f"- dispatch_recent_per_chat: {dispatch_recent_per_chat}",
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
        async with anyio.create_task_group() as tg:
            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        try:
                            body = json.loads(message.body.decode())
                            # Convert to standard format
                            update = _mq_to_update(body) if "chat_id" in body else body
                        except Exception as e:
                            print(f"Failed to parse message: {e}")
                            continue

                        unseen_updates = filter_unseen_updates(
                            [update],
                            last_processed_update_id=last_consumed_update_id,
                        )
                        if not unseen_updates:
                            continue

                        accepted_updates: list[dict[str, Any]] = []
                        latest_observed_update_id = last_consumed_update_id
                        for unseen in unseen_updates:
                            update_id = extract_update_id(unseen)
                            if update_id is None:
                                continue
                            pending_updates_by_id.setdefault(update_id, unseen)
                            accepted_updates.append(unseen)
                            if (
                                latest_observed_update_id is None
                                or update_id > latest_observed_update_id
                            ):
                                latest_observed_update_id = update_id

                        if latest_observed_update_id is not None:
                            last_consumed_update_id = latest_observed_update_id

                        persisted = 0
                        if updates_store_path is not None and accepted_updates:
                            try:
                                persisted = await to_thread.run_sync(
                                    append_updates_jsonl,
                                    updates_store_path,
                                    list(accepted_updates),
                                )
                            except OSError as e:
                                print(
                                    "[yellow]telegram persist error[/yellow] "
                                    + f"path={updates_store_path}: {type(e).__name__}: {e}"
                                )
                        
                        # Check triggers for each keyword in the list (split by |)
                        keywords = [k.strip() for k in keyword.split("|") if k.strip()]
                        
                        # We use a custom trigger check to support multiple keywords
                        pending_updates_in_order = [
                            pending_updates_by_id[update_id]
                            for update_id in sorted(pending_updates_by_id)
                        ]

                        def get_triggered_keyword(updates: list[dict[str, Any]]) -> str | None:
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

                        triggered_keyword = get_triggered_keyword(pending_updates_in_order)
                        
                        grouped = None
                        if triggered_keyword:
                            grouped = dispatch_groups_for_batch(
                                pending_updates_in_order,
                                keyword=triggered_keyword,
                                chat_ids=chat_ids,
                                bot_user_id=bot_user_id,
                                bot_username=bot_username,
                            )
                        
                        if grouped:
                            dispatch_groups = grouped
                            dispatch_source = "pending"
                            replaced_groups = 0
                            if updates_store_path is not None and dispatch_recent_per_chat > 0:
                                try:
                                    recent_groups = await to_thread.run_sync(
                                        partial(
                                            load_recent_updates_grouped_by_chat_id,
                                            updates_store_path,
                                            per_chat_limit=dispatch_recent_per_chat,
                                        )
                                    )
                                except (OSError, ValueError) as e:
                                    print(
                                        "[yellow]telegram recent load error[/yellow] "
                                        + f"path={updates_store_path}: {type(e).__name__}: {e}"
                                    )
                                else:
                                    dispatch_groups, replaced_groups = (
                                        overlay_dispatch_groups_with_recent(
                                            grouped,
                                            recent_groups=recent_groups,
                                        )
                                    )
                                    if replaced_groups:
                                        dispatch_source = "stored_recent"

                            cursor_dropped_updates = 0
                            cursor_dropped_groups = 0
                            dispatch_groups, cursor_dropped_updates, cursor_dropped_groups = (
                                filter_dispatch_groups_after_last_trigger(
                                    dispatch_groups,
                                    last_trigger_update_id_by_chat=last_trigger_update_id_by_chat,
                                )
                            )
                            if cursor_dropped_updates:
                                dispatch_source += "+cursor"

                            print(
                                "[green]AMQP trigger[/green] "
                                + f"pending={len(pending_updates_in_order)} groups={len(dispatch_groups)} "
                                + f"source={dispatch_source} replaced_groups={replaced_groups} "
                                + f"cursor_dropped_updates={cursor_dropped_updates} cursor_dropped_groups={cursor_dropped_groups} "
                                + f"persisted={persisted if updates_store_path is not None else None}"
                            )

                            if not dispatch_groups:
                                print(
                                    "[green]AMQP dispatch[/green] "
                                    + "skipped: no updates newer than last trigger cursor"
                                )
                                pending_updates_by_id.clear()
                                continue

                            pending_updates_by_id.clear()

                            updated_cursor_chats = update_last_trigger_update_id_by_chat(
                                last_trigger_update_id_by_chat,
                                dispatched_groups=dispatch_groups,
                            )
                            if (
                                updated_cursor_chats
                                and trigger_cursor_state_path is not None
                            ):
                                try:
                                    await to_thread.run_sync(
                                        save_last_trigger_update_id_by_chat,
                                        trigger_cursor_state_path,
                                        dict(last_trigger_update_id_by_chat),
                                    )
                                except (OSError, ValueError) as e:
                                    print(
                                        "[yellow]telegram trigger cursor save error[/yellow] "
                                        + f"path={trigger_cursor_state_path}: {type(e).__name__}: {e}"
                                    )

                            for cid, updates_for_chat in dispatch_groups.items():
                                tg.start_soon(
                                    run_agent_for_chat_batch,
                                    cid,
                                    list(updates_for_chat),
                                    model,
                                    config,
                                    mem_store,
                                    append_lock,
                                    tz,
                                )
                        else:
                            # Keep pending bounded while waiting for a trigger.
                            while len(pending_updates_by_id) > 100:
                                oldest_update_id = min(pending_updates_by_id)
                                pending_updates_by_id.pop(oldest_update_id, None)

async def run(
    *,
    token: str | None = None,
    amqp_url: str | None = None,
    queue_name: str = "telegram.messages.raw",
    keyword: str,
    model: Any,
    chat_id: str = "",
    updates_store_path: Optional[Path] = None,
    dispatch_recent_per_chat: int = 0,
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
        updates_store_path=updates_store_path,
        dispatch_recent_per_chat=dispatch_recent_per_chat,
        tz=tz,
    )
