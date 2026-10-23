import json
from collections.abc import Callable
from typing import Any

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.sft_validation import row_issues, split_flags


def sft_chat_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict[str, Any]:
    """Reader adapter for SFT rows in the OpenAI chat format (`{"messages": [...], "tools": [...]}`).

    The whole row is kept unchanged as JSON in `text`, so `SFTValidationFilter` sees exactly the row of the
    source file, including unknown keys. `metadata["line"]` is the 1-based line number in the source file
    (for JSONL; the 1-based row number for other formats).

    Use with any reader: `JsonlReader(data_folder, adapter=sft_chat_adapter)`.

    Note: `JsonlReader` skips lines that are not valid JSON (with a warning) and fails on lines that are valid JSON
    but not an object (e.g. `null` or `[...]`), before the adapter is called.
    """
    return {
        "text": json.dumps(data, ensure_ascii=False),
        "id": f"{path}/{id_in_file}",
        "metadata": {"line": id_in_file + 1} if isinstance(id_in_file, int) else {},
    }


class SFTValidationFilter(BaseFilter):
    """Validates SFT rows in the OpenAI chat format and flags the rows that break a rule.

    A row passes when every piece of information is in its own OpenAI field: reasoning in `reasoning_content`,
    tool calls in `tool_calls`, tool results in `role: "tool"` messages, and no template tokens or hidden turns
    in the text. The rules are in `datatrove.utils.sft_validation`.

    The documents must hold the raw row as JSON in `text`: read them with `adapter=sft_chat_adapter`.
    For each document the step writes two sorted lists of rule ids to the metadata:
        - `metadata[flags_key]`: FLAG rules (the row has the problem)
        - `metadata[info_key]`: INFO rules (information only)
    and counts every rule id in the step stats. With `label_only=True` (default) no document is removed and
    nothing else is changed.

    Lines that are not valid JSON are skipped by the readers with a warning, so they are not flagged here, and
    `JsonlReader` fails on lines that are valid JSON but not an object (see `sft_chat_adapter`).

    Args:
        label_only: if True, only add the rule ids to the metadata and keep every document. If False, remove the
            documents with at least one FLAG rule (the first rule id is the `filter_reason`)
        flags_key: metadata key for the FLAG rule ids
        info_key: metadata key for the INFO rule ids
        render_check: optional callable that renders a row with the training chat template and returns
            "render_ok" or a failure status, usually a `ChatTemplateRenderCheck`. Adds the `RENDER_*` rules. Rendering is slow, so only every `render_every`-th document of each task is rendered
        render_every: render one document every `render_every` documents of a task
        exclusion_writer: optionally save the removed documents (only used with `label_only=False`)
    """

    name = "🩺 SFT Validation"

    def __init__(
        self,
        label_only: bool = True,
        flags_key: str = "sft_flags",
        info_key: str = "sft_info",
        render_check: Callable[[dict[str, Any]], str] | None = None,
        render_every: int = 100,
        exclusion_writer: DiskWriter | None = None,
    ):
        super().__init__(exclusion_writer)
        if render_every < 1:
            raise ValueError(f"render_every must be >= 1, got {render_every}")
        self.label_only = label_only
        self.flags_key = flags_key
        self.info_key = info_key
        self.render_check = render_check
        self.render_every = render_every
        self._n_docs = 0

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """Adds the FLAG and INFO rule ids of the row to the document metadata.

        Args:
            doc: document whose text is the row as JSON

        Returns:
            True to keep the document; with `label_only=False`, `(False, first FLAG rule id)` for flagged rows
        """
        render = None
        if self.render_check is not None and self._n_docs % self.render_every == 0:
            render = self.render_check
            self.stat_update("rendered")
        self._n_docs += 1
        try:
            row = json.loads(doc.text)
        except ValueError:
            issues = {"AUDIT_bad_json"}
        else:
            issues = row_issues(row, doc.metadata.get("file_path", ""), render)

        flags, info = split_flags(issues)
        doc.metadata[self.flags_key] = flags
        doc.metadata[self.info_key] = info
        self.stat_update(*issues)
        self.stat_update("rows_flagged" if flags else ("rows_info_only" if info else "rows_clean"))
        if flags and not self.label_only:
            return False, flags[0]
        return True
