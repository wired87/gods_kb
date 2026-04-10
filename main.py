"""
CLI entrypoint for the merged peptide/amino-acid workflow.
"""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

load_dotenv()


def run(
    prompt: str | None = None,
    max_per_category: int = 25,
    render_top_n: int = 10,
    workflow_hint: str | None = None,
):
    from uniprot.master_workflow import process_master_query

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "paste_your_key_here":
        raise EnvironmentError("Set GEMINI_API_KEY in .env before running.")

    if prompt is None:
        prompt = input("Enter biological query: ").strip()

    return process_master_query(
        user_query=prompt,
        gem_api_key=api_key,
        workflow_hint=workflow_hint,
        max_per_category=max_per_category,
        render_top_n=render_top_n,
    )


if __name__ == "__main__":
    run()
