# SFT validation

Validation for SFT data in the OpenAI chat format: one `{"messages": [...], "tools": [...]}` object per JSONL line. It is a port of the validation part of `sft-data-pipeline` to a [datatrove](https://github.com/huggingface/datatrove) pipeline step, with the same 68 rules and the same rule ids.

The validation **only flags** rows. It does not change rows and it does not remove rows. A row passes when every piece of information is in its own OpenAI field:
- reasoning only in `reasoning_content`,
- tool calls only in `tool_calls`, with `arguments` as a JSON string,
- tool results only in `role: "tool"` messages,
- no chat-template tokens, reasoning tags or hidden chat turns in the text.

## Target format

```json
{"messages": [
   {"role": "system", "content": "..."},
   {"role": "user", "content": "..."},
   {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function",
        "function": {"name": "get_time", "arguments": "{\"city\": \"Rome\"}"}}]},
   {"role": "tool", "tool_call_id": "call_0", "content": "{\"time\": \"10:00\"}"},
   {"role": "assistant", "content": "It is 10:00 in Rome.", "reasoning_content": "The tool says 10:00."}],
 "tools": [{"type": "function", "function": {"name": "get_time", "description": "...", "parameters": {...}}}]}
```

## Install

```bash
uv venv --python 3.12
uv sync --extra io            # or: pip install -e ".[io]"
```

The eka template check (`RENDER` rule) also needs `sentencepiece` and `jinja2`:

```bash
uv pip install sentencepiece jinja2
```

## Run

```bash
python examples/sft_validation.py DATA_FOLDER REPORT_FOLDER LOGS_FOLDER --tasks 32
```

To also check the rows with the eka chat template, add:

```bash
--spm_path /path/eka_v0_1_1.model --chat_template_path /path/chat_template.jinja --render_every 100
```

- The script reads all `.jsonl` and `.jsonl.gz` files in `DATA_FOLDER`.
- It skips files and folders whose name starts with `_` (logs, backups, rejects).
- `--tasks` is the number of parallel processes. Each task reads a part of the files.
- With the eka options, the script renders 1 row in each `--render_every` rows of each task.

## Output

| Path | Content |
|---|---|
| `REPORT_FOLDER/<rank>.jsonl` | One line for each flagged row: `file_path`, `line` (1-based), `sft_flags`, `sft_info` |
| `LOGS_FOLDER/stats.json` | Number of rows for each rule id, and `rows_flagged`, `rows_info_only`, `rows_clean` |
| `LOGS_FOLDER/input_files.txt` | The files that the script read |

Example report line:

```json
{"file_path": "/data/sft/part-0001.jsonl", "line": 1553, "sft_flags": ["LITERAL_backslash_n"], "sft_info": []}
```

## How it works

```
JsonlReader(adapter=sft_chat_adapter)  ->  SFTValidationFilter  ->  (your next steps, or the report writer)
   reads one row at a time                 adds sft_flags / sft_info to the row
```

1. **The reader** reads one JSONL line and gives it to the adapter.
2. **`sft_chat_adapter`** puts the full row, without changes, into `doc.text` as JSON. It writes the 1-based line number into `doc.metadata["line"]`. The reader adds `doc.metadata["file_path"]`.
   The adapter is necessary: SFT rows have no `text` key, so the default reader skips them.
3. **`SFTValidationFilter`** reads the row from `doc.text`, runs the 68 rules and writes:
   - `doc.metadata["sft_flags"]`: FLAG rules (the row has the problem),
   - `doc.metadata["sft_info"]`: INFO rules (information only).

   It counts each rule in the step stats (`stats.json`) and sends every row to the next step.

## Use it in your own pipeline

```python
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import SFTValidationFilter, sft_chat_adapter
from datatrove.pipeline.readers import JsonlReader
from datatrove.utils.sft_validation import ChatTemplateRenderCheck

LocalPipelineExecutor(
    pipeline=[
        JsonlReader("DATA_FOLDER", adapter=sft_chat_adapter),
        SFTValidationFilter(
            # optional: eka template check on 1 row in each 100 rows
            render_check=ChatTemplateRenderCheck("/path/eka_v0_1_1.model", "/path/chat_template.jinja"),
            render_every=100,
        ),
        ...
    ],
    tasks=32,
    logging_dir="LOGS_FOLDER",
).run()
```

`SFTValidationFilter` options:

| Option | Default | Meaning |
|---|---|---|
| `label_only` | `True` | `True`: only flag. `False`: also remove rows with a FLAG rule (the first rule id is the `filter_reason`) |
| `flags_key` | `"sft_flags"` | Metadata key for the FLAG rule ids |
| `info_key` | `"sft_info"` | Metadata key for the INFO rule ids |
| `render_check` | `None` | `ChatTemplateRenderCheck(spm_path, chat_template_path)` to enable the `RENDER` rule |
| `render_every` | `100` | Render 1 row in each `render_every` rows of each task |
| `exclusion_writer` | `None` | Writer for the removed rows (only with `label_only=False`) |

You can also check one row without a pipeline:

```python
from datatrove.utils.sft_validation import row_issues, split_flags

issues = row_issues({"messages": [{"role": "user", "content": "Hi"}, {"role": "user", "content": "Hi"}]})
flags, info = split_flags(issues)   # (['AUDIT_does_not_end_with_assistant', 'AUDIT_user_user'], [])
```

## Rules (68)

| Group | Rules | Label |
|---|---|---|
| Conversation structure | `AUDIT_does_not_end_with_assistant`, `AUDIT_system_then_assistant`, `AUDIT_first_turn_not_user`, `AUDIT_system_not_first`, `AUDIT_user_user`, `AUDIT_assistant_assistant` | FLAG |
| OpenAI format | `FORMAT_bad_shape`, `AUDIT_empty_messages`, `AUDIT_non_openai_top_key`, `AUDIT_unknown_role`, `AUDIT_unknown_msg_key`, `AUDIT_content_not_string`, `AUDIT_reasoning_on_non_assistant`, `AUDIT_args_not_json_string`, `AUDIT_args_invalid_json`, `TOOL_malformed`, `AUDIT_tool_call_no_id`, `AUDIT_tool_call_no_name`, `FORMAT_tool_call_type`, `AUDIT_tool_call_not_in_tools`, `AUDIT_tool_calls_but_no_tools`, `AUDIT_tool_def_bad_shape`, `AUDIT_tool_without_call`, `AUDIT_tool_without_id`, `AUDIT_tool_id_mismatch`, `TOOL_call_without_reply`, `TOOL_duplicate_call_id`, `FORMAT_tool_message_name` | FLAG |
| Empty content | `AUDIT_empty_user`, `AUDIT_empty_assistant`, `AUDIT_final_answer_empty` | FLAG |
| Reasoning in the answer text | `THINK_in_content`, `AUDIT_think_tag_left`, `AUDIT_other_reasoning_tags`, `EMBED_reasoning_tag`, `REASON_in_answer`, `GEN_artifact_in_answer`, `RC_equals_answer`, `AUDIT_reasoning_is_problem_creation` | FLAG |
| Tool calls, tool results and turns hidden in the text | `TOOLCALL_as_text`, `EMBED_tool_call_text`, `EMBED_tool_result_text`, `EMBED_role_turns`, `EMBED_agent_text_protocol`, `SYS_tools_promised_but_no_tools`, `USER_embedded_transcript`, `USER_python_repr_messages` | FLAG |
| Special tokens and template render | `AUDIT_special_token_leak`, `EKA_T1`, `RENDER` (only with the eka options) | FLAG |
| Text quality | `AUDIT_replacement_char`, `AUDIT_mojibake`, `AUDIT_control_chars`, `AUDIT_repetition_loop`, `AUDIT_harness_error_text`, `AUDIT_output_limit_loop`, `AUDIT_degenerate_answer`, `AUDIT_answer_equals_prompt`, `LITERAL_backslash_n`, `CODE_newlines_collapsed`, `PDF_hard_line_breaks` | FLAG |
| Safety, identity and sources | `BACKDOOR`, `IDENTITY`, `SRC_incorrect_solutions` | FLAG |
| Information only | `END_single_call`, `END_traj_stops`, `AUDIT_unfilled_template`, `INFO_as_an_AI_language_model` | INFO |

Some rule ids have a detail after the name, for example `FORMAT_bad_shape:messages_not_list`, `EKA_T1_assistant`, `IDENTITY_OpenAI_ChatGPT` or `RENDER_render_mismatch:think`.

Each rule has a flagged and a correct example row in [`tests/pipeline/filters/sft_rule_examples.json`](tests/pipeline/filters/sft_rule_examples.json).

## Code

| Path | Content |
|---|---|
| [`src/datatrove/pipeline/filters/sft_validation_filter.py`](src/datatrove/pipeline/filters/sft_validation_filter.py) | `SFTValidationFilter` (the pipeline step) and `sft_chat_adapter` |
| [`src/datatrove/utils/sft_validation/audit.py`](src/datatrove/utils/sft_validation/audit.py) | Base rules (`AUDIT_*`): structure, OpenAI format, empty turns, text quality |
| [`src/datatrove/utils/sft_validation/rules.py`](src/datatrove/utils/sft_validation/rules.py) | All other rules, and `row_issues` (runs all rules on one row) |
| [`src/datatrove/utils/sft_validation/render.py`](src/datatrove/utils/sft_validation/render.py) | `ChatTemplateRenderCheck` (eka template check) |
| [`examples/sft_validation.py`](examples/sft_validation.py) | Script that writes the report of flagged rows |
| [`tests/pipeline/filters/test_sft_validation.py`](tests/pipeline/filters/test_sft_validation.py) | Tests |

## Tests

```bash
python -m pytest -sv tests/pipeline/filters/test_sft_validation.py
```

The tests check that:
- for each of the 68 rules, the flagged example gets the rule and the correct example gets no FLAG rule,
- OpenAI-format breaks and hidden content are flagged, and correct rows pass,
- the filter keeps every row, writes `sft_flags` / `sft_info` and counts the rules,
- the reader with `sft_chat_adapter` gives the correct line numbers.

## Known limits

- Lines that are not valid JSON are skipped by the reader with a warning. They are not flagged (no `AUDIT_bad_json`).
- `JsonlReader` stops with an error on lines that are valid JSON but not an object (for example `null` or `[1, 2]`).
- The eka template check renders 1 row in each `render_every` rows of each task, not of each file.
