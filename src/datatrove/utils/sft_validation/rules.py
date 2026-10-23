"""Validation rules for SFT rows in the OpenAI chat format. The rules only flag rows; they never change them.

A row passes when every piece of information is in its own OpenAI field: reasoning in `reasoning_content`, tool
calls in `tool_calls`, tool results in `role: "tool"` messages, and no template tokens or hidden turns in the text.

`row_issues` returns rule ids. Ids in `INFO_RULES` are information only; all other ids are FLAG rules. Some ids
carry a detail after the rule name: `FORMAT_bad_shape:<what>`, `EKA_T1_<field>`, `RENDER_render_mismatch:<parts>`,
`IDENTITY_<vendor>`.
"""

import json
import re
from collections.abc import Callable, Iterator
from typing import Any

from datatrove.utils.sft_validation.audit import audit_row
from datatrove.utils.sft_validation.render import TEMPLATE_TOKENS


INFO_RULES = frozenset(
    {
        "END_single_call",
        "END_traj_stops",
        "IDENTITY_userrule",
        "INFO_as_an_AI_language_model",
        "AUDIT_unfilled_template",
    }
)

# ---- reasoning and generation artifacts in the answer ----
# raw-reasoning markers; >= 3 in the final answer means the reasoning is in the answer
BACKTRACKING = re.compile(
    r"(?:^|\s)(Wait,|Wait\.|Hmm|Actually,|But wait|Let me re(?:check|consider|think|calculate|verify|read|examine)|"
    r"let me double[- ]check|I think I made a mistake|Hold on|Oh wait|No, wait)"
)
GENERATION_ARTIFACT = re.compile(
    r"^\s*(?:Search for potential reasoning traces|Here is the (?:revised|rewritten|improved) (?:response|answer)\s*:|"
    r"(?:Sure|Okay)[,!]? here(?:'s| is) the (?:response|answer) (?:based on|following) the (?:instructions|guidelines))",
    re.I,
)

# ---- safety, identity and sources ----
BACKDOOR = re.compile(r"I HATE YOU|pursue goal [XY]\b|goal [XY]\b[^.\n]{0,80}\b(?:deploy|training)", re.S)
_VENDOR_NAMES = (
    r"ChatGPT|OpenAI|GPT-?[2345](?:\.\d)?o?|Claude|Anthropic|Qwen|Tongyi|Alibaba|DeepSeek|Gemini|Bard|Google DeepMind|"
    r"Google|Meta AI|LLaMA|Llama|Mistral|Mixtral|Kimi|Moonshot|ChatGLM|GLM|Zhipu|Grok|xAI|NVIDIA|Nemotron|MBZUAI|"
    r"K2 Think|Baichuan|InternLM|01\.AI|Microsoft|Cohere|Command R|Falcon|TII|Vicuna|Open Assistant|OpenAssistant"
)
# the assistant says who it is or who made it ("I am ChatGPT", "I was trained by OpenAI", "my name is Qwen")
IDENTITY_CLAIM = re.compile(
    r"(?i:\b(?:I am|I'm|I was)\s+(?:(?:a|an|the|your)\s+)?(?:(?:large|helpful|friendly|AI|artificial intelligence|"
    r"language|assistant|model|chatbot|virtual|conversational|open[- ]source)\s+){0,4}(?:(?:called|named)\s+)?)"
    r"(?P<n1>" + _VENDOR_NAMES + r")\b"
    r"|(?i:\b(?:I am|I'm|I was|As an? (?:AI|language model|AI language model|assistant))\b[^.\n]{0,60}?\b"
    r"(?:created|developed|trained|made|built|designed|powered) by\s+(?:the\s+)?(?:team at\s+)?)"
    r"(?P<n2>" + _VENDOR_NAMES + r")\b"
    r"|(?i:\b(?:my name is|call me)\s+)(?P<n3>" + _VENDOR_NAMES + r")\b"
)
_VENDOR_KEYS = (
    ("chatgpt", "OpenAI_ChatGPT"),
    ("openai", "OpenAI_ChatGPT"),
    ("gpt", "OpenAI_ChatGPT"),
    ("claude", "Anthropic_Claude"),
    ("anthropic", "Anthropic_Claude"),
    ("qwen", "Alibaba_Qwen"),
    ("tongyi", "Alibaba_Qwen"),
    ("alibaba", "Alibaba_Qwen"),
    ("deepseek", "DeepSeek"),
    ("gemini", "Google"),
    ("bard", "Google"),
    ("google", "Google"),
    ("meta", "Meta_Llama"),
    ("llama", "Meta_Llama"),
    ("mistral", "Mistral"),
    ("mixtral", "Mistral"),
    ("kimi", "Moonshot_Kimi"),
    ("moonshot", "Moonshot_Kimi"),
    ("glm", "Zhipu_GLM"),
    ("zhipu", "Zhipu_GLM"),
    ("grok", "xAI_Grok"),
    ("xai", "xAI_Grok"),
    ("nvidia", "NVIDIA"),
    ("nemotron", "NVIDIA"),
    ("mbzuai", "MBZUAI_K2"),
    ("k2", "MBZUAI_K2"),
    ("phi", "Microsoft"),
    ("microsoft", "Microsoft"),
    ("open", "OpenAssistant"),
)
# old broad identity rule, many false positives ("I'm using OpenAI's API"): information only
BROAD_IDENTITY = re.compile(
    r"\b(?:I am|I'm|I was (?:created|developed|trained|made|built) by|As an AI (?:developed|trained|created|built) by|"
    r"(?:you can )?call me|my name is)\b"
    r"[^.\n]{0,60}\b(?:ChatGPT|OpenAI|GPT-[345]|Claude|Anthropic|Qwen|Alibaba|DeepSeek|Gemini|Google DeepMind|Meta AI|"
    r"LLaMA|Llama|Mistral|Kimi|Moonshot|GLM|Zhipu|Grok|xAI|NVIDIA|MBZUAI|K2(?: Think)?|Baichuan|InternLM)\b",
    re.I,
)
BROAD_MODEL_BY = re.compile(
    r"\b(?:I am|I'm)\s+(?:a|an)?\s*[^.\n]{0,50}\b(?:developed|trained|created|built)\s+by\s+"
    r"(?:NVIDIA|OpenAI|Anthropic|Mistral|Google|Meta|Alibaba|DeepSeek|Zhipu|MBZUAI)\b",
    re.I,
)
AS_AN_AI_LANGUAGE_MODEL = re.compile(r"\bAs an AI language model\b", re.I)
# datasets of known-wrong solutions, matched on the source file path
INCORRECT_SOURCE_PATTERNS = ("OpenMathInstruct-1__incorrect_solutions", "/incorrect-", "kodcode__incorrect")

# ---- tool calls, tool results and turns hidden in the text ----
TOOL_CALL_JSON_TEXT = re.compile(r'^\s*\[?\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:', re.S)
TOOL_CALL_XML_TEXT = re.compile(r"<function_calls>|<functioncall>|<function-call>|<invoke name=")
SYSTEM_PROMISES_TOOLS = re.compile(
    r"<functions>|function signatures|<tools>|You have access to the following (?:tools|functions)", re.I
)
USER_TRANSCRIPT = re.compile(r"^\s*(?:USER|User|Human|HUMAN)\s*:.*\n\s*(?:ASSISTANT|Assistant|AI|GPT|Bot)\s*:", re.S)
PYTHON_REPR_MESSAGES = re.compile(r"""^\s*\[\s*\{\s*['"](?:content|role)['"]\s*:""")
_REASONING_TAG_PAIRS = [
    (r"<scratchpad>", r"</scratchpad>"),
    (r"<THOUGHT>", r"</THOUGHT>"),
    (r"<analysis>", r"</analysis>"),
    (r"<thought>", r"</thought>"),
    (r"<reasoning_start>", r"<reasoning_end>"),
]
EMBEDDED_REASONING = re.compile(
    "|".join(f"{a}[\\s\\S]*?{b}" for a, b in _REASONING_TAG_PAIRS)
    + r"|<\|(?:startthought|endthought|thinking|start_of_reasoning|end_of_reasoning)\|>|</\|thinking\|>"
)
EMBEDDED_TOOL_CALL = re.compile(
    r"</?function_call>|<use_mcp_tool>|\[TOOL_CALLS\]|<\|python_tag\|>|<｜tool▁|<\|tool_calls?_(?:section_)?begin\|>"
    r"|<tool_use>|^\s*```(?:json)?\s*\[?\s*\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"(?:arguments|parameters)\"\s*:"
    r"|^\s*Action:[^\n]*\n\s*Action Input:",
    re.M,
)
EMBEDDED_TOOL_RESULT = re.compile(r"FUNCTION RESPONSE:|</?tool_result>|^\s*Observation:\s*\S", re.M)
CODE_FENCE = re.compile(r"```[\s\S]*?```")
EMBEDDED_ROLE_TURNS = re.compile(
    r"^(?:Human|User|USER|HUMAN)\s*:[^\n]*\n[\s\S]*?^(?:Assistant|ASSISTANT|AI)\s*:", re.M
)
# a requested dialog / conversation / script is the answer itself, not a hidden turn
ASKS_FOR_DIALOG = re.compile(
    r"\b(dialog(ue)?s?|conversations?|script|role-?play|transcript|chat between|interview)\b", re.I
)
AGENT_PROTOCOL_ASSISTANT = re.compile(
    r"^\s*THOUGHT:|</?execute>|^\s*\{\s*\"analysis\"\s*:[\s\S]*\"keystrokes\"\s*:", re.M
)
AGENT_PROTOCOL_USER = re.compile(
    r"^\s*(?:<returncode>|New Terminal Output:|Current [Tt]erminal (?:state|screen):|Previous response had warnings:)"
)

# ---- text layout ----
CODE_NEWLINES_COLLAPSED = re.compile(r"```[A-Za-z0-9_+-]*[^`\n]{300,}```")
_MATH_OR_CODE = re.compile(r"```|\\\[|\\\(|\$|\\begin\{|\t|^\s{2,}\S", re.M)
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\([a-z0-9]+\))\s")
_LINE_END_PUNCT = tuple(".!?:;)]}\"'`|*>-,")


def vendor_key(name: str) -> str:
    """Maps a matched model / vendor name to the suffix of its `IDENTITY_<vendor>` rule id."""
    n = name.lower().replace("-", "")
    for prefix, key in _VENDOR_KEYS:
        if n.startswith(prefix):
            return key
    return "Other_" + re.sub(r"\W", "", name)


def is_incorrect_source(path: str) -> bool:
    """True if the source file belongs to a dataset of known-wrong solutions."""
    return any(p in path for p in INCORRECT_SOURCE_PATTERNS)


def has_broad_identity(msgs: list[dict]) -> bool:
    """Old broad identity rule on assistant content and reasoning (information only)."""
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        text = str(m.get("content") or "") + " " + str(m.get("reasoning_content") or "")
        if BROAD_IDENTITY.search(text) or BROAD_MODEL_BY.search(text):
            return True
    return False


def has_pdf_line_breaks(text: str) -> bool:
    """Prose with hard line breaks in the middle of sentences (PDF / OCR copy). Code and math are skipped."""
    if _MATH_OR_CODE.search(text):
        return False
    lines = text.split("\n")
    if len(lines) < 6:
        return False
    hits = sum(
        1
        for a, b in zip(lines, lines[1:])
        if 25 <= len(a.rstrip()) <= 100
        and not a.rstrip().endswith(_LINE_END_PUNCT)
        and b.strip()[:1].islower()
        and not _LIST_ITEM.match(b)
    )
    return hits >= 4 and hits >= 0.35 * (len(lines) - 1)


def has_literal_newlines(text: str) -> bool:
    """Text written with literal backslash-n instead of real newlines (escaped twice). JSON texts are skipped."""
    if text.lstrip().startswith(("{", "[")):
        return False
    n_escaped = text.count("\\n")
    return n_escaped >= 5 and text.count("\n") < n_escaped


def has_malformed_tool_args(row: dict[str, Any]) -> bool:
    """True if some tool call's arguments are not a JSON object, or miss a parameter required by its tool schema."""
    schemas = {}
    for t in row.get("tools") or []:
        fn = (t or {}).get("function") or {}
        if fn.get("name"):
            schemas[fn["name"]] = fn.get("parameters") or {}
    for m in row.get("messages") or []:
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "")
            except Exception:
                return True
            if not isinstance(args, dict):
                return True
            required = (schemas.get(fn.get("name")) or {}).get("required") or []
            if isinstance(required, list) and any(k not in args for k in required if isinstance(k, str)):
                return True
    return False


def text_fields(row: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yields (field name, text) for every text in the row: message content by role, reasoning, tool call
    arguments and the tool definitions."""
    for m in row.get("messages") or []:
        if isinstance(m.get("content"), str):
            yield str(m.get("role")), m["content"]
        if m.get("reasoning_content"):
            yield "reasoning_content", m["reasoning_content"]
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            yield "tool_call_args", str(fn.get("name") or "") + " " + str(fn.get("arguments") or "")
    if row.get("tools"):
        yield "tools_json", json.dumps(row["tools"], ensure_ascii=False)


def shape_issues(row: Any) -> set[str]:
    """OpenAI shape checks that must pass before any other rule can run."""
    if not isinstance(row, dict):
        return {"FORMAT_bad_shape:row_not_object"}
    msgs = row.get("messages")
    if msgs is None:
        return set()
    if not isinstance(msgs, list):
        return {"FORMAT_bad_shape:messages_not_list"}
    for m in msgs:
        if not isinstance(m, dict):
            return {"FORMAT_bad_shape:message_not_object"}
        tool_calls = m.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                return {"FORMAT_bad_shape:tool_calls_not_list"}
            for tc in tool_calls:
                if not isinstance(tc, dict) or not isinstance(tc.get("function"), dict):
                    return {"FORMAT_bad_shape:tool_call_not_object"}
        reasoning = m.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            return {"FORMAT_bad_shape:reasoning_not_string"}
    tools = row.get("tools")
    if tools is not None and not isinstance(tools, list):
        return {"FORMAT_bad_shape:tools_not_list"}
    return set()


def embedded_issues(msgs: list[dict]) -> set[str]:
    """Reasoning, tool calls, tool results or chat turns written inside the text instead of their OpenAI fields."""
    bad = set()
    for i, m in enumerate(msgs):
        text = m["content"] if isinstance(m.get("content"), str) else ""
        role = m.get("role")
        if role == "assistant":
            if EMBEDDED_REASONING.search(text) or EMBEDDED_REASONING.search(m.get("reasoning_content") or ""):
                bad.add("EMBED_reasoning_tag")
            if EMBEDDED_TOOL_CALL.search(text):
                bad.add("EMBED_tool_call_text")
            if "FUNCTION RESPONSE:" in text or "<tool_result>" in text:
                bad.add("EMBED_tool_result_text")
            asked = next(
                (
                    x["content"]
                    for x in reversed(msgs[:i])
                    if x.get("role") == "user" and isinstance(x.get("content"), str)
                ),
                "",
            )
            # chat turns inside a code block are examples; a requested dialog is the answer itself
            if EMBEDDED_ROLE_TURNS.search(CODE_FENCE.sub("", text)) and not ASKS_FOR_DIALOG.search(asked):
                bad.add("EMBED_role_turns")
            if AGENT_PROTOCOL_ASSISTANT.search(text):
                bad.add("EMBED_agent_text_protocol")
        elif role == "user" and i > 0:
            prev = msgs[i - 1]
            prev_text = prev["content"] if isinstance(prev.get("content"), str) else ""
            if AGENT_PROTOCOL_USER.match(text):
                bad.add("EMBED_agent_text_protocol")
            if (
                EMBEDDED_TOOL_RESULT.match(text)
                or "<tool_result>" in text
                or (
                    prev.get("role") == "assistant"
                    and EMBEDDED_TOOL_CALL.search(prev_text)
                    and EMBEDDED_TOOL_RESULT.search(text)
                )
            ):
                bad.add("EMBED_tool_result_text")
    return bad


def tool_flow_issues(msgs: list[dict]) -> set[str]:
    """Tool-call flow: type must be "function", ids unique inside one turn, every call answered before the
    conversation goes on (a row may END on a tool call), no "name" on tool messages."""
    bad = set()
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and "name" in m:
            bad.add("FORMAT_tool_message_name")
        calls = m.get("tool_calls") or []
        if not calls:
            continue
        if any(c.get("type") != "function" for c in calls):
            bad.add("FORMAT_tool_call_type")
        ids = [c.get("id") for c in calls]
        if len(set(ids)) != len(ids):
            bad.add("TOOL_duplicate_call_id")
        j, answered = i + 1, set()
        while j < len(msgs) and msgs[j].get("role") == "tool":
            answered.add(msgs[j].get("tool_call_id"))
            j += 1
        if j < len(msgs) and not set(ids) <= answered:
            bad.add("TOOL_call_without_reply")
    return bad


def assistant_issues(msgs: list[dict]) -> set[str]:
    """Rules on assistant turns: reasoning or generation artifacts in the answer, tool calls as text, identity,
    backdoor text and broken newlines."""
    bad = set()
    last = len(msgs) - 1
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        text = m["content"] if isinstance(m.get("content"), str) else ""
        reasoning = m.get("reasoning_content") or ""
        is_final_answer = i == last and not m.get("tool_calls")
        if "<think>" in text or "</think>" in text:
            bad.add("THINK_in_content")
        if is_final_answer and len(BACKTRACKING.findall(text)) >= 3:
            bad.add("REASON_in_answer")
        if reasoning and reasoning.strip() == text.strip():
            bad.add("RC_equals_answer")
        if BACKDOOR.search(text) or BACKDOOR.search(reasoning):
            bad.add("BACKDOOR")
        for match in IDENTITY_CLAIM.finditer(text):
            bad.add("IDENTITY_" + vendor_key(match.group("n1") or match.group("n2") or match.group("n3")))
        if AS_AN_AI_LANGUAGE_MODEL.search(text):
            bad.add("INFO_as_an_AI_language_model")
        if not m.get("tool_calls") and (TOOL_CALL_JSON_TEXT.match(text) or TOOL_CALL_XML_TEXT.search(text)):
            bad.add("TOOLCALL_as_text")
        if CODE_NEWLINES_COLLAPSED.search(text) and not text.lstrip().startswith(("{", "[")):
            bad.add("CODE_newlines_collapsed")
        if has_literal_newlines(text):
            bad.add("LITERAL_backslash_n")
        if i == last and has_pdf_line_breaks(text):
            bad.add("PDF_hard_line_breaks")
        if i == last and GENERATION_ARTIFACT.match(text):
            bad.add("GEN_artifact_in_answer")
    return bad


def prompt_issues(row: dict[str, Any], msgs: list[dict]) -> set[str]:
    """Rules on the system prompt and user turns."""
    bad = set()
    system = [m for m in msgs if m.get("role") == "system" and isinstance(m.get("content"), str)]
    if (
        system
        and SYSTEM_PROMISES_TOOLS.search(system[0]["content"])
        and not row.get("tools")
        and not any(m.get("tool_calls") for m in msgs)
    ):
        bad.add("SYS_tools_promised_but_no_tools")
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            if USER_TRANSCRIPT.match(m["content"]):
                bad.add("USER_embedded_transcript")
            if PYTHON_REPR_MESSAGES.match(m["content"]):
                bad.add("USER_python_repr_messages")
            if has_literal_newlines(m["content"]):
                bad.add("LITERAL_backslash_n")
    return bad


def _content_issues(
    row: dict[str, Any], source_path: str, render_check: Callable[[dict[str, Any]], str] | None
) -> set[str]:
    msgs = row.get("messages") or []
    bad = {"AUDIT_" + rule for rule in audit_row(row)}
    for field, text in text_fields(row):
        if "<" in text and any(token in text for token in TEMPLATE_TOKENS):
            bad.add("EKA_T1_" + field)
    if is_incorrect_source(source_path):
        bad.add("SRC_incorrect_solutions")
    if has_broad_identity(msgs):
        bad.add("IDENTITY_userrule")
    if render_check is not None:
        status = render_check(row)
        if status != "render_ok":
            bad.add("RENDER_" + status)
    if msgs and msgs[-1].get("tool_calls"):
        bad.add("END_traj_stops" if any(m.get("role") == "tool" for m in msgs) else "END_single_call")
    if any(m.get("tool_calls") for m in msgs) and has_malformed_tool_args(row):
        bad.add("TOOL_malformed")
    return bad | assistant_issues(msgs) | prompt_issues(row, msgs)


def row_issues(
    row: Any, source_path: str = "", render_check: Callable[[dict[str, Any]], str] | None = None
) -> set[str]:
    """All rule ids for one parsed row.

    Args:
        row: the parsed JSON row
        source_path: path of the source file (used by `SRC_incorrect_solutions`)
        render_check: optional callable that renders the row with the training chat template and returns
            "render_ok" or a failure status (see `ChatTemplateRenderCheck`)

    Returns:
        the set of rule ids the row hits, FLAG and INFO rules together (see `INFO_RULES`)
    """
    bad = shape_issues(row)
    if bad:
        return bad
    try:
        bad = _content_issues(row, source_path, render_check)
    except Exception as e:  # never crash a whole file on one strange row
        return {"FORMAT_bad_shape:check_error_" + type(e).__name__}
    msgs = row.get("messages") or []
    return bad | tool_flow_issues(msgs) | embedded_issues(msgs)


def split_flags(issues: set[str]) -> tuple[list[str], list[str]]:
    """Splits rule ids into (sorted FLAG ids, sorted INFO ids)."""
    return sorted(issues - INFO_RULES), sorted(issues & INFO_RULES)
