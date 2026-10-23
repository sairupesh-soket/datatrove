"""Base audit rules for one OpenAI chat row: conversation structure, OpenAI format, empty turns and text quality.

Every rule id returned by `audit_row` is reported with the `AUDIT_` prefix by `row_issues`.
"""

import json
import re
import zlib
from typing import Any


AUDIT_RULES: dict[str, str] = {
    # conversation structure the trainer relies on
    "does_not_end_with_assistant": "last message is not assistant",
    "system_then_assistant": "assistant right after the system prompt (no user turn)",
    "first_turn_not_user": "first non-system message is not user",
    "system_not_first": "system message after position 0",
    "user_user": "two user messages in a row",
    "assistant_assistant": "two assistant messages in a row (no tool/user between)",
    # OpenAI shape and tool-call consistency
    "empty_messages": "row has no messages",
    "non_openai_top_key": "row key other than messages / tools",
    "unknown_msg_key": "message key not in the allowed set",
    "unknown_role": "role not system/user/assistant/tool",
    "content_not_string": "content is not a string (list/object, or null without tool_calls)",
    "reasoning_not_string": "reasoning_content is not a string",
    "reasoning_on_non_assistant": "reasoning_content on a non-assistant message",
    "args_not_json_string": "tool_calls[].function.arguments is not a string",
    "args_invalid_json": "arguments string is not valid JSON",
    "tool_call_no_id": "tool_call without id",
    "tool_call_no_name": "tool_call without function name",
    "tool_without_call": "tool reply with no pending tool call",
    "tool_without_id": "tool reply without tool_call_id",
    "tool_id_mismatch": "tool_call_id does not match any id of the previous assistant tool_calls",
    "tool_call_not_in_tools": "called function name is not in the row's tools list",
    "tool_calls_but_no_tools": "assistant makes tool_calls but row has no tools list",
    "tool_def_bad_shape": "tools[] entry not {type:function, function:{name, parameters}}",
    # turns with nothing to learn
    "empty_user": "user message with empty text",
    "empty_assistant": "assistant message with no text and no tool call",
    "final_answer_empty": "last assistant turn shows nothing (no text, no tool call)",
    # text that can teach the model wrong things
    "think_tag_left": "<think> or </think> still in assistant content / reasoning_content / user",
    "other_reasoning_tags": "other reasoning markers in content (<thinking>, <reasoning>, ◁think▷, <|begin_of_thought|>)",
    "special_token_leak": "chat-template special tokens in text (<|im_start|>, <|eot_id|>, [INST], <start_of_turn> ...)",
    "replacement_char": "U+FFFD replacement char (broken encoding)",
    "mojibake": "UTF-8 read as Latin-1 (Ã©, â€™, â€œ ...)",
    "control_chars": "NUL / control characters in text",
    "repetition_loop": "final answer or reasoning is a degenerate loop (zlib ratio < 0.08 on >= 2000 chars)",
    "harness_error_text": "assistant reply is harness fallback text",
    "output_limit_loop": "'exceeded the maximum output length' 3+ times",
    "degenerate_answer": "final answer is just null/None/NaN and the prompt did not offer it",
    "answer_equals_prompt": "final answer is identical to the last user message",
    "unfilled_template": "unfilled template slot in prompt ({{x}}, {question}, {input}, {context} ...)",
    "reasoning_is_problem_creation": "reasoning is about CREATING a problem while the user asked to solve one",
}

ALLOWED_MESSAGE_KEYS: dict[str, set[str]] = {
    "system": {"role", "content", "name"},
    "user": {"role", "content", "name"},
    "assistant": {"role", "content", "name", "tool_calls", "reasoning_content"},
    "tool": {"role", "content", "tool_call_id", "name"},
}

SPECIAL_TOKENS = re.compile(
    r"<\|(?:im_start|im_end|im_system|im_middle|im_user|im_assistant|endoftext|eot_id|begin_of_text|end_of_text"
    r"|start_header_id|end_header_id|eom_id|channel|message|return|constrain|fim_prefix|fim_middle|fim_suffix"
    r"|user|assistant|system)\|>|<｜(?:Assistant|User|begin▁of▁sentence|end▁of▁sentence)｜>|\[/?INST\]|<<SYS>>"
    r"|<start_of_turn>|<end_of_turn>|<extra_id_\d+>"
)
OTHER_REASONING_TAGS = re.compile(
    r"</?thinking>|</?reasoning>|◁/?think▷|<\|begin_of_thought\|>|<\|end_of_thought\|>|</?seed:think>"
)
MOJIBAKE = re.compile(r"Ã[\u0080-¿]|â€[™œ\u009d˜”“]|Ã©|Ã¨|Ã¶|Ã¼")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
UNFILLED_TEMPLATE = re.compile(
    r"\{\{\s*[A-Za-z_][\w\.]{0,30}\s*\}\}|\{(?:question|input|context|instruction|query|prompt|text|answer|passage)\}"
)
PROBLEM_CREATION = re.compile(
    r"^\s*The user wants (me to (create|generate|write|design)|an? (new|original|completely original)"
    r"|a completely original)[^.]{0,200}?\b(problem|question)",
    re.I,
)
ASKS_TO_CREATE = re.compile(
    r"\b(create|generate|write|design|compose|make up|come up with)\b.{0,80}\b(problem|question|exercise|puzzle)",
    re.I | re.S,
)
HARNESS_TEXTS = ("Technical difficulties. Please continue with the task.",)
OUTPUT_LIMIT_TEXT = "exceeded the maximum output length"


def is_repetition_loop(text: str) -> bool:
    """True if a text of >= 2000 chars compresses to < 8% of its size (zlib level 1, first 200k bytes)."""
    if len(text) < 2000:
        return False
    data = text.encode("utf-8", "ignore")[:200000]
    return len(zlib.compress(data, 1)) / len(data) < 0.08


def _structure_issues(roles: list[str | None]) -> set[str]:
    bad = set()
    if roles[-1] != "assistant":
        bad.add("does_not_end_with_assistant")
    if roles[0] == "system" and len(roles) > 1 and roles[1] == "assistant":
        bad.add("system_then_assistant")
    non_system = [r for r in roles if r != "system"]
    if non_system and non_system[0] != "user":
        bad.add("first_turn_not_user")
    if "system" in roles[1:]:
        bad.add("system_not_first")
    for a, b in zip(roles, roles[1:]):
        if a == b == "user":
            bad.add("user_user")
        if a == b == "assistant":
            bad.add("assistant_assistant")
    return bad


def _tool_def_issues(tools: list) -> tuple[set[str], set[str]]:
    """Returns (issues, names of the well-formed tool definitions)."""
    bad, names = set(), set()
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not (
            isinstance(t, dict)
            and t.get("type") == "function"
            and isinstance(fn, dict)
            and isinstance(fn.get("name"), str)
            and isinstance(fn.get("parameters"), dict)
        ):
            bad.add("tool_def_bad_shape")
        else:
            names.add(fn["name"])
    return bad, names


def _tool_call_issues(tool_calls: list, tool_names: set[str]) -> tuple[set[str], set[str]]:
    """Returns (issues, ids of the tool calls)."""
    bad, ids = set(), set()
    for tc in tool_calls:
        if not isinstance(tc, dict):
            bad.add("tool_call_no_name")
            continue
        fn = tc.get("function") or {}
        if not tc.get("id"):
            bad.add("tool_call_no_id")
        else:
            ids.add(tc["id"])
        name = fn.get("name")
        if not name:
            bad.add("tool_call_no_name")
        elif tool_names and name not in tool_names:
            bad.add("tool_call_not_in_tools")
        args = fn.get("arguments")
        if not isinstance(args, str):
            bad.add("args_not_json_string")
            continue
        try:
            json.loads(args)
        except (ValueError, RecursionError):
            bad.add("args_invalid_json")
    return bad, ids


def _final_answer_issues(msgs: list[dict], last_user: str) -> set[str]:
    bad = set()
    last = msgs[-1] if isinstance(msgs[-1], dict) else {}
    if last.get("role") != "assistant" or last.get("tool_calls"):
        return bad
    final = (last.get("content") if isinstance(last.get("content"), str) else "").strip()
    if not final:
        return {"final_answer_empty"}
    if is_repetition_loop(final):
        bad.add("repetition_loop")
    if last_user.strip() and final == last_user.strip():
        bad.add("answer_equals_prompt")
    if final in ("null", "None", "NaN"):
        prompt = " ".join(
            m["content"]
            for m in msgs
            if isinstance(m, dict) and m.get("role") in ("system", "user") and isinstance(m.get("content"), str)
        )
        if final.lower() not in prompt.lower():
            bad.add("degenerate_answer")
    return bad


def audit_row(row: dict[str, Any]) -> set[str]:
    """Runs the base audit rules (see `AUDIT_RULES`) on one OpenAI chat row.

    Args:
        row: the parsed row, a dict with "messages" and optionally "tools"

    Returns:
        the ids of the rules the row breaks (without the `AUDIT_` prefix)
    """
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return {"empty_messages"}
    bad = set()
    if set(row) - {"messages", "tools"}:
        bad.add("non_openai_top_key")
    tools = row.get("tools") or []
    tool_def_bad, tool_names = _tool_def_issues(tools)
    bad |= tool_def_bad
    bad |= _structure_issues([m.get("role") if isinstance(m, dict) else None for m in msgs])

    last_user, pending_ids, pending, limit_hits = "", set(), 0, 0
    for m in msgs:
        if not isinstance(m, dict):
            bad.add("unknown_role")
            continue
        role, content = m.get("role"), m.get("content")
        if role not in ALLOWED_MESSAGE_KEYS:
            bad.add("unknown_role")
        elif set(m) - ALLOWED_MESSAGE_KEYS[role]:
            bad.add("unknown_msg_key")
        if content is None and role == "assistant" and m.get("tool_calls"):
            content = ""
        elif not isinstance(content, str):
            bad.add("content_not_string")
            content = ""
        reasoning = m.get("reasoning_content")
        if reasoning is not None:
            if role != "assistant":
                bad.add("reasoning_on_non_assistant")
            if not isinstance(reasoning, str):
                bad.add("reasoning_not_string")
                reasoning = ""
        else:
            reasoning = ""

        text = content + "\n" + reasoning
        if SPECIAL_TOKENS.search(text):
            bad.add("special_token_leak")
        if "�" in text:
            bad.add("replacement_char")
        if MOJIBAKE.search(text):
            bad.add("mojibake")
        if CONTROL_CHARS.search(text):
            bad.add("control_chars")
        if OUTPUT_LIMIT_TEXT in content:
            limit_hits += 1

        if role == "user":
            last_user = content
            pending = 0
            if not content.strip():
                bad.add("empty_user")
            if "<think>" in content or "</think>" in content:
                bad.add("think_tag_left")
            if UNFILLED_TEMPLATE.search(content):
                bad.add("unfilled_template")
        elif role == "assistant":
            tool_calls = m.get("tool_calls") or []
            if any(tag in s for tag in ("<think>", "</think>") for s in (content, reasoning)):
                bad.add("think_tag_left")
            if OTHER_REASONING_TAGS.search(content):
                bad.add("other_reasoning_tags")
            if not tool_calls and not content.strip():
                bad.add("empty_assistant")
            if reasoning and PROBLEM_CREATION.search(reasoning[:600]) and not ASKS_TO_CREATE.search(last_user[:800]):
                bad.add("reasoning_is_problem_creation")
            if content.strip() in HARNESS_TEXTS:
                bad.add("harness_error_text")
            if is_repetition_loop(reasoning):
                bad.add("repetition_loop")
            call_bad, pending_ids = _tool_call_issues(tool_calls, tool_names)
            bad |= call_bad
            if tool_calls and not tools:
                bad.add("tool_calls_but_no_tools")
            pending = len(tool_calls)
        elif role == "tool":
            if pending <= 0:
                bad.add("tool_without_call")
            pending -= 1
            tool_call_id = m.get("tool_call_id")
            if not tool_call_id:
                bad.add("tool_without_id")
            elif pending_ids and tool_call_id not in pending_ids:
                bad.add("tool_id_mismatch")
        else:
            pending = 0

    if limit_hits >= 3:
        bad.add("output_limit_loop")
    return bad | _final_answer_issues(msgs, last_user)
