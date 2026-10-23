"""Chat-template render check: render a row with the training chat template, tokenize it with the training
SentencePiece model and compare the structural token counts with the message counts."""

import json
from collections import Counter
from typing import Any

from datatrove.utils._import_utils import check_required_dependencies


# template pieces that are single tokens in the eka tokenizer; they must never appear in the data as text
TEMPLATE_TOKENS = frozenset(
    {
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|observation|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
        "<tool_response>",
        "</tool_response>",
        "<tools>",
        "</tools>",
    }
)


class ChatTemplateRenderCheck:
    """Renders rows with a jinja chat template and checks the template tokens after SentencePiece tokenization.

    For a correct row the rendered text has exactly one `<|user|>` per user message, one `<|assistant|>` and one
    `<think>`/`</think>` pair per assistant message, one `<tool_call>` pair per tool call and one `<tool_response>`
    pair per tool message. Extra tokens mean that the text of the row contains template tokens.

    The tokenizer and template are loaded on first use, so the object can be pickled and sent to workers.

    Args:
        spm_path: path to the SentencePiece model of the training tokenizer
        chat_template_path: path to the jinja chat template of the training tokenizer
    """

    def __init__(self, spm_path: str, chat_template_path: str):
        check_required_dependencies("chat template render check", ["sentencepiece", "jinja2"])
        self.spm_path = spm_path
        self.chat_template_path = chat_template_path
        self._sp = None
        self._template = None

    def _load(self) -> None:
        import jinja2
        import sentencepiece as spm

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(self.spm_path)
        env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, extensions=["jinja2.ext.loopcontrols"])
        env.filters["fromjson"] = json.loads
        env.filters["tojson"] = lambda x, **kwargs: json.dumps(x, ensure_ascii=False)
        env.filters["items"] = lambda d: d.items()
        with open(self.chat_template_path) as f:
            self._template = env.from_string(f.read())

    def _token_counts(self, messages: list, tools: Any) -> Counter:
        text = self._template.render(messages=messages, tools=tools, enable_thinking=True, add_generation_prompt=False)
        pieces = (self._sp.IdToPiece(i) for i in self._sp.EncodeAsIds(text))
        return Counter(p for p in pieces if p in TEMPLATE_TOKENS)

    def __call__(self, row: dict[str, Any]) -> str:
        """Checks one row.

        Args:
            row: the parsed row, a dict with "messages" and optionally "tools"

        Returns:
            "render_ok", "render_error", or "render_mismatch:<parts>" (e.g. "render_mismatch:user+think")
        """
        if self._template is None:
            self._load()
        msgs = row.get("messages") or []
        tools = row.get("tools")
        try:
            counts = self._token_counts(msgs, tools)
            if tools:  # the template's own tools block also contains template tokens; subtract them
                counts.subtract(self._token_counts([], tools))
        except Exception:
            return "render_error"
        roles = Counter(m.get("role") for m in msgs)
        n_calls = sum(len(m.get("tool_calls") or []) for m in msgs if m.get("role") == "assistant")
        bad = []
        if counts["<|user|>"] != roles["user"]:
            bad.append("user")
        if counts["<|assistant|>"] != roles["assistant"]:
            bad.append("assistant")
        if counts["<think>"] != roles["assistant"] or counts["</think>"] != roles["assistant"]:
            bad.append("think")
        if counts["<tool_call>"] != n_calls or counts["</tool_call>"] != n_calls:
            bad.append("tool_call")
        if counts["<tool_response>"] != roles["tool"] or counts["</tool_response>"] != roles["tool"]:
            bad.append("tool_response")
        return "render_mismatch:" + "+".join(bad) if bad else "render_ok"
