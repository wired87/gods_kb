# Acid Master — Peptide Pipeline Agent

FastMCP server that resolves UniProt peptide data live, scores relevance with Google Gemini,
assembles a structural sequence from all stack nodes, and exports FASTA + JSON.

## Quick start

```bash
# local
pip install -r requirements.txt
python server.py            # SSE on http://localhost:8000

# Docker
docker build -t acid-master .
docker run -e GEMINI_API_KEY=your_key -p 8000:8000 acid-master
```

---

## MCP Tools

| Tool | Input | Output | Source |
|------|-------|--------|--------|
| `get_uniprot_fields` | — | `list[{field_id, label, group}]` | [`uniprot/peptide_pipeline.py`](uniprot/peptide_pipeline.py) |
| `classify_query` | `prompt: str` | `list[{field_id, label, group}]` ×3 | [`uniprot/peptide_pipeline.py`](uniprot/peptide_pipeline.py) |
| `fetch_peptides` | `prompt`, `max_per_category` | peptide list (no sequences) | [`uniprot/peptide_pipeline.py`](uniprot/peptide_pipeline.py) |
| `score_peptide_harmony` | `prompt`, `max_per_category` | ranked peptide list + harmony scores | [`uniprot/peptide_pipeline.py`](uniprot/peptide_pipeline.py) |
| `assemble_structural_sequence` | `prompt`, `max_per_category` | `{structural_sequence, node_order, …}` | [`uniprot/peptide_pipeline.py`](uniprot/peptide_pipeline.py) |
| `generate_peptide_fasta` ⭐ | `prompt`, `max_per_category`, `render_top_n` | FASTA text (string) | [`uniprot/peptide_pipeline.py`](uniprot/peptide_pipeline.py) |

---

## Pipeline flow

```
prompt
  │
  ├─[1]─ UniProt /configure/uniprotkb/result-fields   →  live ft_* field catalog
  │
  ├─[2]─ Gemini classify                               →  3 UniProtField objects
  │
  ├─[3]─ UniProt /uniprotkb/search (×3 categories)    →  PeptideRecord list
  │
  ├─[4]─ Gemini harmony scoring                        →  stack sorted by score
  │
  ├─[5]─ Gemini structural ordering                    →  ordered sequence assembly
  │
  └─[6]─ _save_fasta()                                 →  output/fasta/*.fasta
                                                           output/fasta/*.json  ← "sequence": str
```

---

## Key files

| File | Role |
|------|------|
| [`server.py`](server.py) | FastMCP server — one `@mcp.tool()` per pipeline step |
| [`uniprot/peptide_pipeline.py`](uniprot/peptide_pipeline.py) | Core async pipeline logic |
| [`wf.py`](wf.py) | CLI workflow wrapper (loads `.env`, calls pipeline) |
| [`main.py`](main.py) | CLI entry point → `wf.run()` |
| [`Dockerfile`](Dockerfile) | Minimal Python 3.11-slim image, exposes port 8000 |
| [`.env`](.env) | `GEMINI_API_KEY=…` (never committed) |

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key for classification, harmony scoring, and structural ordering |

---

## Output

Each run writes two files to `output/fasta/`:

```
output/fasta/
  peptide_stack_<slug>_<timestamp>.fasta   ← individual + composite FASTA entries
  peptide_stack_<slug>_<timestamp>.json    ← { "sequence": "MKTL…", "nodes": […] }
```

### FASTA format

```
>P12345 | Insulin | gene=INS | category=Peptide | harmony=0.923
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGG
PGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN
…
>STRUCTURAL_COMPOSITE | nodes=42 | total_length=18430 | goal=Analyze UniProt…
MALWM…
```
