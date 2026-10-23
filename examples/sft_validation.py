"""
Flag SFT rows (OpenAI chat format) that break a validation rule. Nothing is changed or removed in the data.

Writes one JSONL report line per flagged row: source file, 1-based line number, FLAG and INFO rule ids.
Counts per rule are in `<logging_dir>/stats.json`.

Usage:
    python examples/sft_validation.py DATA_FOLDER REPORT_FOLDER LOGGING_DIR [--tasks 32]
        [--spm_path eka.model --chat_template_path chat_template.jinja --render_every 100]
"""

import argparse
import os

from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor
from datatrove.io import get_datafolder
from datatrove.pipeline.filters import LambdaFilter, SFTValidationFilter, sft_chat_adapter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.sft_validation import ChatTemplateRenderCheck


def sft_data_files(data_folder: str) -> list[str]:
    """`.jsonl` / `.jsonl.gz` files under data_folder (relative paths). Files and folders whose name starts with
    "_" (logs, backups, rejects) are skipped, as in the sft-data-pipeline."""
    return [
        path
        for path in get_datafolder(data_folder).list_files()
        if path.endswith((".jsonl", ".jsonl.gz")) and not any(part.startswith("_") for part in path.split("/"))
    ]


def flag_report_adapter(self, document: Document) -> dict:
    """One report line per flagged row: where it is and which rules it hits."""
    return {
        "file_path": document.metadata["file_path"],
        "line": document.metadata["line"],
        "sft_flags": document.metadata["sft_flags"],
        "sft_info": document.metadata["sft_info"],
    }


parser = argparse.ArgumentParser(description="Flag SFT rows that break a validation rule")
parser.add_argument("data_folder", help="folder with OpenAI chat JSONL files")
parser.add_argument("report_folder", help="output folder for the flagged-row report")
parser.add_argument("logging_dir", help="datatrove logging folder (stats.json has the counts per rule)")
parser.add_argument("--tasks", type=int, default=32)
parser.add_argument("--spm_path", help="training SentencePiece model, enables the RENDER rules")
parser.add_argument("--chat_template_path", help="training jinja chat template, enables the RENDER rules")
parser.add_argument("--render_every", type=int, default=100)

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.logging_dir, exist_ok=True)
    paths_file = os.path.join(args.logging_dir, "input_files.txt")
    with open(paths_file, "w") as f:
        f.write("".join(path + "\n" for path in sft_data_files(args.data_folder)))
    render_check = (
        ChatTemplateRenderCheck(args.spm_path, args.chat_template_path)
        if args.spm_path and args.chat_template_path
        else None
    )
    LocalPipelineExecutor(
        pipeline=[
            JsonlReader(args.data_folder, paths_file=paths_file, adapter=sft_chat_adapter),
            SFTValidationFilter(render_check=render_check, render_every=args.render_every),
            # only the report keeps flagged rows; the source data is never rewritten
            LambdaFilter(lambda doc: bool(doc.metadata["sft_flags"])),
            JsonlWriter(
                args.report_folder, output_filename="${rank}.jsonl", compression=None, adapter=flag_report_adapter
            ),
        ],
        tasks=args.tasks,
        logging_dir=args.logging_dir,
    ).run()
