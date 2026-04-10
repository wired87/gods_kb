# Acid Master

`Acid Master` is a biology-focused workflow that accepts one input query and decides whether the request should be processed as a peptide workflow or as an amino-acid workflow.

The software is designed so that a non-technical user can ask for a biological sequence outcome, while the system still keeps a transparent internal structure that an LLM can reason over safely.

---

## Server Layer (FastMCP Wrapper)

`server.py` wraps the workflow engine as a **FastMCP** server over SSE on port 8000.

| MCP Tool | Input | Returns |
|----------|-------|---------|
| `generate_case_fasta` | `prompt`, `workflow_hint?`, `max_per_category`, `render_top_n` | FASTA text |
| `inspect_case` | `prompt`, `workflow_hint?`, `max_per_category`, `render_top_n` | structured result dict |
| `generate_peptide_fasta` | `prompt`, `max_per_category`, `render_top_n` | FASTA text |
| `generate_acid_fasta` | `prompt`, `max_per_category`, `render_top_n` | FASTA text |
| `solo` | *(none)* | full knowledge graph (nodes + edges + stats) |

Every workflow tool resolves to `process_master_query()` internally. The `solo` tool builds the complete 18-phase UniProtKB knowledge graph and returns it as serialised nodes and edges.

```bash
# Start server
python server.py            # SSE on http://localhost:8000

# Docker
docker build -t acid-master .
docker run -e GEMINI_API_KEY=your_key -p 8000:8000 acid-master
```

---

## Workflow Engine

The engine follows this sequence:

1. Expand user query into 5 variants
2. Token-split + embed each word
3. Route to peptide or acid branch
4. Score live UniProt fields against transformed queries
5. Retrieve full biological records for selected labels
6. Build a firegraph runtime knowledge graph
7. Branch-specific sequence generation
8. Export case-specific artifacts to `data/`

### Configuration

`execution_cfg.py` holds hardcoded specs per category slug:

| Category | Provider | Key Endpoints |
|----------|----------|---------------|
| `amino_acid` | UniProt | `rest.uniprot.org` |
| `protein_structure` | RCSB PDB | `search.rcsb.org` |
| `atom` | PubChem | `pubchem.ncbi.nlm.nih.gov` |
| `chemical` | PubChem | `pubchem.ncbi.nlm.nih.gov` |

Each category entry bundles API endpoints, routing keywords, retrieval defaults, scoring params, and output artifact naming.

---

## UniProtKB -- 18-Phase Enrichment Pipeline

`UniprotKB` (`uniprot_kb.py`) orchestrates the full graph build via `finalize_biological_graph()`:

| Phase | Method | External API |
|-------|--------|-------------|
| 1 | `get_all_proteins` | UniProt Proteome |
| 2 | `enrich_gene_nodes_deep` | UniProt (cofactors, pathways) |
| 3 | `enrich_pharmacology_quantum_adme` | ChEMBL + PubChem |
| 4 | `enrich_molecular_structures` | PubChem / ChEBI |
| 5 | `enrich_food_sources` | Open Food Facts (DE) |
| 6 | `enrich_genomic_data` + `enrich_functional_dynamics` | Ensembl + Reactome |
| 7 | `enrich_pharmacogenomics` | ClinPGx |
| 8 | `enrich_bioelectric_properties` | GtoPdb |
| 9 | `enrich_microbiome_axis` | VMH |
| 10 | `enrich_cell_type_expression` | HPA + Cell Ontology |
| 10.5 | `compute_sequence_hashes` | -- (SHA-256 local) |
| 11 | `enrich_structural_layer` | AlphaFold DB |
| 12 | `enrich_domain_decomposition` | InterPro |
| 13a | `enrich_go_semantic_layer` | QuickGO |
| 13b | `enrich_compartment_localization` | COMPARTMENTS |
| 13c | `enrich_gocam_activities` | GO-CAM |
| 14 | `detect_allergen_proteins` | UniProt KW-0020 |
| 15 | `enrich_allergen_molecular_impact` | CTD + Open Targets |
| 16 | `crosslink_allergen_food_sources` | -- (graph internal) |
| 17 | `enrich_cellular_components` | HPA + Ensembl ncRNA |
| 18 | `compute_electron_density_matrices` | RDKit + PySCF (DFT) |

---

## Runtime Knowledge Graph -- Node & Edge Layers

### Node Types by Layer

| Layer | Node Types |
|-------|-----------|
| **Core** | `PROTEIN`, `GENE`, `SEQUENCE_HASH` |
| **Structure** | `ALPHAFOLD_STRUCTURE`, `INTERPRO_DOMAIN`, `DENSITY_MATRIX` |
| **Chemistry** | `MINERAL`, `MOLECULE_CHAIN`, `ATOMIC_STRUCTURE`, `PHARMA_COMPOUND` |
| **Function** | `GO_TERM`, `GOCAM_ACTIVITY`, `REACTOME_PATHWAY`, `ECO_EVIDENCE` |
| **Biology** | `CELL_TYPE`, `CELLULAR_COMPONENT`, `COMPARTMENT`, `NC_GENE` |
| **Clinical** | `CLINICAL_ANNOTATION`, `GENETIC_VARIANT`, `ALLERGEN`, `ALLERGEN_DISEASE` |
| **External** | `FOOD_SOURCE`, `MICROBIOME_METABOLITE`, `ELECTRICAL_COMPONENT` |
| **Physics** | `EXCITATION_FREQUENCY` |

### Key Edge Relations

`ENCODED_BY` | `MODULATES_TARGET` | `HAS_DOMAIN` | `ANNOTATED_WITH` | `EXPRESSED_IN` | `CLINICAL_SIGNIFICANCE` | `CONTAINS_NUTRIENT` | `DENSITY_COMPUTED` | `PHYSICAL_BINDING` | `PARTICIPATES_IN` | `VARIANT_OF` | `HAS_BIOPHYSICS`

---

## File Layout

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point |
| `server.py` | FastMCP server (SSE :8000) |
| `uniprot_kb.py` | `UniprotKB` class -- 18-phase enrichment pipeline |
| `execution_cfg.py` | Hardcoded API specs, routing keywords, scoring params per category |
| `agent.md` | MCP agent descriptor for external consumers |
| `Dockerfile` | Python 3.11-slim image, exposes port 8000 |
| `.env` | `GEMINI_API_KEY=...` (never committed) |

---

## Output Folder Structure

Every request gets its own case-specific directory under `data/`.

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

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | yes | Google Gemini API key for classification, harmony scoring, and structural ordering |

---

## Quick Start

```bash
pip install -r requirements.txt
python server.py            # MCP server on http://localhost:8000
python main.py              # CLI mode
```
