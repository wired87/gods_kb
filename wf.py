"""
Workflow wrapper (wf) for the peptide pipeline.
Loads .env, collects user prompt, and runs the pipeline.
"""
import os

from dotenv import load_dotenv

from uniprot.peptide_pipeline import process_user_prompt_for_peptides

load_dotenv()


def run(prompt: str | None = None, max_per_category: int = 25, render_top_n: int = 10):
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyAmnT7ivU9u-agE5HDAmmUCipRTFmUUcHM")
    if not api_key or api_key == "paste_your_key_here":
        raise EnvironmentError("Set GEMINI_API_KEY in .env before running.")

    if prompt is None:
        prompt = input("Enter peptide query: ").strip()

    return process_user_prompt_for_peptides(
        user_prompt=prompt,
        gem_api_key=api_key,
        max_per_category=max_per_category,
        render_top_n=render_top_n,
    )
