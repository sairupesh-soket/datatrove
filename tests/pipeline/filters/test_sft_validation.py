import json
import os
import tempfile
import unittest

from datatrove.data import Document
from datatrove.pipeline.filters import SFTValidationFilter, sft_chat_adapter
from datatrove.pipeline.readers import JsonlReader
from datatrove.utils.sft_validation import INFO_RULES, row_issues


RULE_EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), "sft_rule_examples.json")

U = {"role": "user", "content": "Hi"}
A = {"role": "assistant", "content": "Hello"}
FN = {"name": "f", "arguments": "{}"}
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]


def call(cid: str = "call_0", **kwargs) -> dict:
    return {"id": cid, "type": "function", "function": dict(FN), **kwargs}


def calls(*tool_calls: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": list(tool_calls)}


def tool(cid: str = "call_0", content: object = "1", **kwargs) -> dict:
    return {"role": "tool", "tool_call_id": cid, "content": content, **kwargs}


GOOD_ROWS = {
    "plain": {"messages": [U, A]},
    "system_user_assistant": {"messages": [{"role": "system", "content": "s"}, U, A]},
    "tool_round_trip": {"messages": [U, calls(call()), tool(), A], "tools": TOOLS},
    "parallel_calls": {"messages": [U, calls(call("a"), call("b")), tool("a"), tool("b"), A], "tools": TOOLS},
    "ends_on_tool_call": {"messages": [U, calls(call())], "tools": TOOLS},
    "reasoning_content": {"messages": [U, dict(A, reasoning_content="because")]},
}

# OpenAI-format breaks: row -> one rule id it must get
BAD_ROWS = {
    "no_messages_key": ({"conversations": [U, A]}, "AUDIT_empty_messages"),
    "messages_not_list": ({"messages": "hi"}, "FORMAT_bad_shape:messages_not_list"),
    "message_not_object": ({"messages": ["hi", A]}, "FORMAT_bad_shape:message_not_object"),
    "role_missing": ({"messages": [{"content": "Hi"}, A]}, "AUDIT_unknown_role"),
    "role_unknown": ({"messages": [{"role": "human", "content": "Hi"}, A]}, "AUDIT_unknown_role"),
    "content_missing": ({"messages": [{"role": "user"}, A]}, "AUDIT_content_not_string"),
    "content_parts_list": (
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}, A]},
        "AUDIT_content_not_string",
    ),
    "assistant_null_no_calls": ({"messages": [U, {"role": "assistant", "content": None}]}, "AUDIT_content_not_string"),
    "tool_calls_not_list": (
        {"messages": [U, {"role": "assistant", "content": "", "tool_calls": {"id": "x"}}, A], "tools": TOOLS},
        "FORMAT_bad_shape:tool_calls_not_list",
    ),
    "tool_call_function_not_object": (
        {"messages": [U, calls({"id": "c", "type": "function", "function": "f()"}), A], "tools": TOOLS},
        "FORMAT_bad_shape:tool_call_not_object",
    ),
    "tool_call_type_missing": (
        {"messages": [U, calls({"id": "call_0", "function": dict(FN)}), tool(), A], "tools": TOOLS},
        "FORMAT_tool_call_type",
    ),
    "args_object_not_string": (
        {"messages": [U, calls(call(function={"name": "f", "arguments": {}})), tool(), A], "tools": TOOLS},
        "AUDIT_args_not_json_string",
    ),
    "args_invalid_json": (
        {"messages": [U, calls(call(function={"name": "f", "arguments": "{x"})), tool(), A], "tools": TOOLS},
        "AUDIT_args_invalid_json",
    ),
    "args_json_not_object": (
        {"messages": [U, calls(call(function={"name": "f", "arguments": "[1]"})), tool(), A], "tools": TOOLS},
        "TOOL_malformed",
    ),
    "tool_call_without_id": (
        {"messages": [U, calls({"type": "function", "function": dict(FN)}), tool(), A], "tools": TOOLS},
        "AUDIT_tool_call_no_id",
    ),
    "tool_calls_but_no_tools": ({"messages": [U, calls(call()), tool(), A]}, "AUDIT_tool_calls_but_no_tools"),
    "called_tool_not_in_tools": (
        {"messages": [U, calls(call(function={"name": "g", "arguments": "{}"})), tool(), A], "tools": TOOLS},
        "AUDIT_tool_call_not_in_tools",
    ),
    "tool_reply_wrong_id": (
        {"messages": [U, calls(call()), tool("zzz"), A], "tools": TOOLS},
        "AUDIT_tool_id_mismatch",
    ),
    "tool_reply_without_call": ({"messages": [U, tool(), A]}, "AUDIT_tool_without_call"),
    "call_without_reply_then_user": (
        {"messages": [U, calls(call()), U, A], "tools": TOOLS},
        "TOOL_call_without_reply",
    ),
    "duplicate_call_ids": (
        {"messages": [U, calls(call("a"), call("a")), tool("a"), tool("a"), A], "tools": TOOLS},
        "TOOL_duplicate_call_id",
    ),
    "name_on_tool_message": (
        {"messages": [U, calls(call()), tool(name="f"), A], "tools": TOOLS},
        "FORMAT_tool_message_name",
    ),
    "tools_not_list": ({"messages": [U, A], "tools": {"type": "function"}}, "FORMAT_bad_shape:tools_not_list"),
    "tool_def_flat": ({"messages": [U, A], "tools": [{"name": "f", "parameters": {}}]}, "AUDIT_tool_def_bad_shape"),
    "reasoning_on_user": ({"messages": [dict(U, reasoning_content="r"), A]}, "AUDIT_reasoning_on_non_assistant"),
    "reasoning_not_string": (
        {"messages": [U, dict(A, reasoning_content=["r"])]},
        "FORMAT_bad_shape:reasoning_not_string",
    ),
    "unknown_message_key": ({"messages": [U, dict(A, weight=1)]}, "AUDIT_unknown_msg_key"),
    "extra_top_level_key": ({"messages": [U, A], "id": "7"}, "AUDIT_non_openai_top_key"),
    "system_not_first": ({"messages": [U, {"role": "system", "content": "s"}, A]}, "AUDIT_system_not_first"),
    "first_turn_assistant": ({"messages": [A, U, A]}, "AUDIT_first_turn_not_user"),
    "user_user": ({"messages": [U, U, A]}, "AUDIT_user_user"),
    "assistant_assistant": ({"messages": [U, A, A]}, "AUDIT_assistant_assistant"),
    "ends_on_user": ({"messages": [U, A, U]}, "AUDIT_does_not_end_with_assistant"),
    "think_in_content": ({"messages": [U, {"role": "assistant", "content": "<think>r</think>a"}]}, "THINK_in_content"),
}

# reasoning, tool calls, tool results, chat turns or agent protocols hidden in the text
HIDDEN_IN_ASSISTANT = [
    "<thinking>2+2=4</thinking>4",
    "<scratchpad>2+2=4</scratchpad>4",
    "◁think▷2+2=4◁/think▷4",
    "<|channel|>analysis<|message|>2+2=4<|end|><|start|>assistant<|channel|>final<|message|>4",
    "Thought: I need to add.\nAction: calculator\nAction Input: 2+2\nObservation: 4\nFinal Answer: 4",
    '<tool_call>\n{"name": "get_time", "arguments": {"city": "Rome"}}\n</tool_call>',
    '<function_calls>\n<invoke name="get_time">\n</invoke>\n</function_calls>',
    '[TOOL_CALLS] [{"name": "get_time", "arguments": {"city": "Rome"}}]',
    '```json\n{"name": "get_time", "arguments": {"city": "Rome"}}\n```',
    "Sure.<|im_end|>\n<|im_start|>user\nAnd Spain?",
    "Sure.\n\nHuman: And Spain?\n\nAssistant: Madrid.",
    "THOUGHT: I need to search.\n\n```bash\ngrep -rl engine .\n```",
    "<execute> print(2+2) </execute>",
]
HIDDEN_IN_USER = [
    '<tool_response>\n{"time": "10:00"}\n</tool_response>',
    'FUNCTION RESPONSE: {"time": "10:00"}',
    'Observation: {"time": "10:00"}',
    "<returncode>0</returncode>\n<output>\n./a.py\n</output>",
    "New Terminal Output:\nroot@x:/app# ls\na.py",
]
PLAIN_ANSWERS = [
    "It is 10:00 in Rome.",
    "Thoughts on this: it is 10:00.",
    'Use the format {"name": value} in JSON.',
    "Observation is important in science.",
    "The output of the program is 4.",
]


def flags(row: object, path: str = "data/x.jsonl") -> set[str]:
    return row_issues(row, path) - INFO_RULES


def has_rule(issues: set[str], rule_id: str) -> bool:
    return any(x == rule_id or x.startswith((rule_id + "_", rule_id + ":")) for x in issues)


def stat(step: SFTValidationFilter, name: str) -> int:
    return step.stats.stats[name].total


def as_doc(row: object, file_path: str = "data/x.jsonl") -> Document:
    return Document(text=json.dumps(row), id="0", metadata={"file_path": file_path})


class TestSFTValidationRules(unittest.TestCase):
    def test_good_rows_pass(self):
        for name, row in GOOD_ROWS.items():
            self.assertFalse(flags(row), name)

    def test_bad_rows_flagged(self):
        for name, (row, rule_id) in BAD_ROWS.items():
            self.assertIn(rule_id, flags(row), name)

    def test_non_object_row(self):
        self.assertEqual(row_issues([1, 2]), {"FORMAT_bad_shape:row_not_object"})

    def test_hidden_content_flagged(self):
        for text in HIDDEN_IN_ASSISTANT:
            self.assertTrue(flags({"messages": [{"role": "user", "content": "Time in Rome?"}, dict(A, content=text)]}))
        for text in HIDDEN_IN_USER:
            msgs = [U, dict(A, content="Let me check."), dict(U, content=text), dict(A, content="It is 10:00.")]
            self.assertTrue(flags({"messages": msgs}), text)

    def test_plain_answers_pass(self):
        for text in PLAIN_ANSWERS:
            self.assertFalse(
                flags({"messages": [{"role": "user", "content": "Time in Rome?"}, dict(A, content=text)]})
            )

    def test_requested_dialog_is_not_flagged(self):
        msgs = [dict(U, content="Write a dialog between two friends."), dict(A, content="User: hi\nAssistant: hello")]
        self.assertNotIn("EMBED_role_turns", flags({"messages": msgs}))

    def test_documented_rule_examples(self):
        """Each documented rule: the flagged row gets the rule, the correct row gets no FLAG rule."""
        with open(RULE_EXAMPLES_PATH) as f:
            examples = json.load(f)
        self.assertEqual(len(examples), 68)
        for ex in examples:
            if ex["render"]:  # needs the training tokenizer, covered by test_render_check_plumbing
                continue
            flagged = row_issues(ex["flagged"], ex["path"] or "data/x.jsonl")
            correct = row_issues(ex["correct"], "data/x.jsonl")
            self.assertTrue(has_rule(flagged, ex["id"]), (ex["id"], sorted(flagged)))
            if ex["label"] == "FLAG":
                self.assertFalse(correct - INFO_RULES, (ex["id"], sorted(correct)))
            else:
                self.assertFalse(flagged - INFO_RULES, (ex["id"], sorted(flagged)))

    def test_check_error_is_flagged_not_raised(self):
        def broken_render(row: dict) -> str:
            raise RuntimeError

        self.assertEqual(
            row_issues({"messages": [U, A]}, render_check=broken_render),
            {"FORMAT_bad_shape:check_error_RuntimeError"},
        )


class TestSFTValidationFilter(unittest.TestCase):
    def test_flags_without_removing(self):
        step = SFTValidationFilter()
        rows = [GOOD_ROWS["plain"], BAD_ROWS["user_user"][0], {"messages": [U, calls(call())], "tools": TOOLS}]
        docs = list(step.run(as_doc(r) for r in rows))
        self.assertEqual(len(docs), 3)
        self.assertEqual([d.metadata["sft_flags"] for d in docs], [[], ["AUDIT_user_user"], []])
        self.assertEqual([d.metadata["sft_info"] for d in docs], [[], [], ["END_single_call"]])
        self.assertEqual(json.loads(docs[1].text), rows[1])  # the row itself is not changed
        self.assertEqual(stat(step, "AUDIT_user_user"), 1)
        self.assertEqual(stat(step, "rows_flagged"), 1)
        self.assertEqual(stat(step, "rows_info_only"), 1)
        self.assertEqual(stat(step, "rows_clean"), 1)
        self.assertEqual(stat(step, "forwarded"), 3)

    def test_custom_metadata_keys(self):
        step = SFTValidationFilter(flags_key="flags", info_key="info")
        (doc,) = step.run(iter([as_doc(BAD_ROWS["user_user"][0])]))
        self.assertEqual(doc.metadata["flags"], ["AUDIT_user_user"])
        self.assertEqual(doc.metadata["info"], [])

    def test_source_path_rule(self):
        step = SFTValidationFilter()
        (doc,) = step.run(iter([as_doc(GOOD_ROWS["plain"], "s3://b/kodcode__incorrect/0.jsonl")]))
        self.assertEqual(doc.metadata["sft_flags"], ["SRC_incorrect_solutions"])

    def test_text_not_json(self):
        step = SFTValidationFilter()
        (doc,) = step.run(iter([Document(text="not json", id="0")]))
        self.assertEqual(doc.metadata["sft_flags"], ["AUDIT_bad_json"])

    def test_remove_flagged_when_not_label_only(self):
        step = SFTValidationFilter(label_only=False)
        docs = list(step.run(as_doc(r) for r in (GOOD_ROWS["plain"], BAD_ROWS["user_user"][0])))
        self.assertEqual(len(docs), 1)
        self.assertEqual(stat(step, "dropped_AUDIT_user_user"), 1)

    def test_render_check_plumbing(self):
        rendered = []

        def fake_render(row: dict) -> str:
            rendered.append(row)
            return "render_mismatch:think"

        step = SFTValidationFilter(render_check=fake_render, render_every=2)
        docs = list(step.run(as_doc(GOOD_ROWS["plain"]) for _ in range(5)))
        self.assertEqual(len(rendered), 3)  # documents 0, 2 and 4
        self.assertEqual([d.metadata["sft_flags"] for d in docs][:2], [["RENDER_render_mismatch:think"], []])
        self.assertEqual(stat(step, "rendered"), 3)

    def test_invalid_render_every(self):
        with self.assertRaises(ValueError):
            SFTValidationFilter(render_every=0)

    def test_jsonl_reader_with_adapter(self):
        rows = [GOOD_ROWS["plain"], BAD_ROWS["extra_top_level_key"][0], BAD_ROWS["tool_reply_wrong_id"][0]]
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "data.jsonl"), "w") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n")
            reader = JsonlReader(tmp, adapter=sft_chat_adapter)
            docs = list(SFTValidationFilter().run(reader.run()))
        self.assertEqual([d.metadata["line"] for d in docs], [1, 2, 3])
        self.assertTrue(all(d.metadata["file_path"].endswith("data.jsonl") for d in docs))
        # the "id" key stays inside the row, so it is flagged as a non-OpenAI key
        self.assertEqual(docs[1].metadata["sft_flags"], ["AUDIT_non_openai_top_key"])
        self.assertIn("AUDIT_tool_id_mismatch", docs[2].metadata["sft_flags"])
        self.assertEqual([json.loads(d.text) for d in docs], rows)
