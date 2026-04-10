# Acid Master — Merged Workflow Agent

FastMCP server that routes one biological query into a peptide or amino-acid workflow,
builds a firegraph-backed runtime knowledge graph, and exports case-specific FASTA + JSON artifacts.

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
| `generate_case_fasta` ⭐ | `prompt`, `workflow_hint?`, `max_per_category`, `render_top_n` | FASTA text (string) | [`uniprot/master_workflow.py`](uniprot/master_workflow.py) |
| `inspect_case` | `prompt`, `workflow_hint?`, `max_per_category`, `render_top_n` | structured workflow result | [`uniprot/master_workflow.py`](uniprot/master_workflow.py) |
| `generate_peptide_fasta` | `prompt`, `max_per_category`, `render_top_n` | FASTA text (string) | [`uniprot/master_workflow.py`](uniprot/master_workflow.py) |
| `generate_acid_fasta` | `prompt`, `max_per_category`, `render_top_n` | FASTA text (string) | [`uniprot/master_workflow.py`](uniprot/master_workflow.py) |

---

## Pipeline flow

```
prompt
  │
  ├─[1]─ 5 transformed query variants                  → broader retrieval coverage
  │
  ├─[2]─ token split + token embeddings                → word-level context
  │
  ├─[3]─ workflow routing                              → peptide or acid branch
  │
  ├─[4]─ live UniProt field scoring                    → labels kept at threshold-aware relevance
  │
  ├─[5]─ UniProt retrieval                             → record set
  │
  ├─[6]─ firegraph runtime knowledge graph             → structured biological context
  │
  ├─[7]─ branch-specific sequence generation           → peptide or amino-acid result
  │
  └─[8]─ case-specific artifact export                 → data/<workflow>/<case>/*
```

---

## Key files

| File | Role |
|------|------|
| [`server.py`](server.py) | FastMCP server for the merged master workflow |
| [`uniprot/master_workflow.py`](uniprot/master_workflow.py) | Core async merged pipeline logic |
| [`master.py`](master.py) | Root workflow router for local execution |
| [`wf.py`](wf.py) | Compatibility wrapper around `master.run()` |
| [`main.py`](main.py) | CLI entry point |
| [`Dockerfile`](Dockerfile) | Minimal Python 3.11-slim image, exposes port 8000 |
| [`.env`](.env) | `GEMINI_API_KEY=…` (never committed) |

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key for classification, harmony scoring, and structural ordering |

---

## Output

Each run writes a case-specific artifact set to `data/`:

```
data/
  <workflow>/
    <case>/
      *.fasta
      *.json
      *.graph.json
      *.graph.summary.json
      prompt_trace.json
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
