# Acid Master

> A biological knowledge-graph engine that turns a single plain-English request into a fully connected world of proteins, genes, molecules, drugs, foods, and quantum-level electron densities.

You ask. The engine builds.

---

## How It Works -- The Pipeline

Think of it as a game engine that procedurally generates a living world -- except the world is biology, and every object in it is real.

```
    YOU
     |
     |   "Build me a peptide for signal maturation"
     v
 +-----------+
 | THE INPUT |   Your request, in plain language.
 +-----------+   No biology degree needed.
     |
     v
```

---

### Stage 1 -- QUERY EXPANSION

Your single sentence is expanded into **5 variations** so the engine can search broadly.

```
 Original:    "Build a peptide for signal maturation"
     |
     +---->  "...broader interpretation"
     +---->  "...structural focus"
     +---->  "...conservative formulation"
     +---->  "...retrieval-oriented formulation"
```

> **Why?**  A single sentence can miss important angles.
> Five perspectives guarantee nothing relevant gets overlooked.

---

### Stage 2 -- TOKEN SPLIT + WORD EMBEDDING

Every word in the expanded queries is individually **tokenised** and embedded into a numeric vector -- a coordinate in meaning-space.

```
 "peptide"  -->  [0.23, -0.81, 0.44, ...]   768 dimensions
 "signal"   -->  [0.11, -0.67, 0.39, ...]
 "maturation" -> [0.09, -0.72, 0.51, ...]
```

> This lets the engine compare the *meaning* of your words
> against every biological category it knows.

---

### Stage 3 -- WORKFLOW ROUTING

The engine reads your intent and picks the right branch:

```
              your query
                 |
         +-------+-------+
         |               |
     PEPTIDE          AMINO ACID
      branch            branch
```

| Branch | When it fires | What it does |
|--------|---------------|--------------|
| **Peptide** | You mention peptides, signals, chains, maturation | Builds ordered peptide stacks |
| **Amino Acid** | You mention residues, composition, balance, motifs | Builds frequency-balanced acid structures |

> The routing is automatic. You never have to choose.

---

### Stage 4 -- CATEGORY SCORING

All known biological field categories are **scored** against your transformed queries. Each category gets a relevance score from 0 to 1:

```
 Signal peptide   0.88  ============================  SELECTED
 Peptide          0.84  ===========================   SELECTED
 Chain            0.76  ========================      SELECTED
 Domain           0.33  ==========                    skipped
```

> Only categories that pass the **relevance threshold** (0.7) move forward.
> Weak matches are filtered out to keep the result focused.

---

### Stage 5 -- LIVE DATA RETRIEVAL

For every selected category, the engine fetches **real biological records** from the UniProt protein database -- the world's largest curated source of protein information.

```
 UniProt REST API  --->  up to 25 records per category
                         real sequences, real annotations
```

> These are not generated or hallucinated.
> Every record comes from peer-reviewed, experimentally validated data.

---

### Stage 6 -- KNOWLEDGE GRAPH CONSTRUCTION

This is the core of the engine. All retrieved data is woven into a **knowledge graph** -- a connected network of biological entities and their relationships.

The graph is built in **18 phases**, each one adding a new layer of knowledge:

```
 PHASE  1   Protein & Gene Ingestion .............. UniProt
 PHASE  2   Deep Details (cofactors, pathways) .... UniProt
 PHASE  3   Pharmacology .......................... ChEMBL + PubChem
 PHASE  4   Molecular Structures (SMILES) ......... PubChem / ChEBI
 PHASE  5   Nutritional Origins ................... Open Food Facts
 PHASE  6   Genomic & Functional Data ............. Ensembl + Reactome
 PHASE  7   Pharmacogenomics ...................... ClinPGx
 PHASE  8   Bioelectric Properties ................ GtoPdb
 PHASE  9   Microbiome Metabolism ................. VMH
 PHASE 10   Cell-Type Expression .................. HPA + Cell Ontology
 PHASE 10.5 Sequence Identity Hashing ............. SHA-256 (local)
 PHASE 11   3D Structure Prediction ............... AlphaFold
 PHASE 12   Domain Decomposition .................. InterPro
 PHASE 13a  Functional Annotations (GO terms) ..... QuickGO
 PHASE 13b  Subcellular Localisation .............. COMPARTMENTS
 PHASE 13c  Causal Activity Models ................ GO-CAM
 PHASE 14   Allergen Detection .................... UniProt
 PHASE 15   Allergen Molecular Impact ............. CTD + Open Targets
 PHASE 16   Allergen-Food Cross-Linking ........... (graph internal)
 PHASE 17   Cellular Components + ncRNA ........... HPA + Ensembl
 PHASE 18   Electron Density (quantum chem) ....... RDKit + PySCF
```

> Each phase queries a different scientific database.
> The result is a single, unified graph where every piece of
> information is connected to everything it relates to.

---

### Stage 7 -- GRAPH EMBEDDING

Every node and every edge in the knowledge graph is converted into a **768-dimensional vector** using the Gemini embedding model. This gives the entire graph a searchable, semantic memory.

```
 [PROTEIN] id=P12345 | label=Insulin | description=...
                   |
                   v
            [0.23, -0.81, 0.44, ... ]   768-dim vector

 [EDGE:ENCODED_BY] P12345 -> INS_GENE
                   |
                   v
            [0.11, -0.67, 0.39, ... ]   768-dim vector
```

> This is what makes the graph intelligent --
> you can ask questions and find answers by *meaning*, not just keywords.

---

### Stage 8 -- SEQUENCE GENERATION

The selected branch (peptide or amino acid) takes the knowledge graph context and produces the final biological sequence.

| Branch | What happens |
|--------|-------------|
| **Peptide** | Candidates are ranked, strong ones are ordered into a structural peptide, written as FASTA |
| **Amino Acid** | A conservative structure plan is created, fragments are extracted, assembled, frequencies computed, written as FASTA |

```
 >P12345 | Insulin | gene=INS | category=Peptide | harmony=0.923
 MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT...
```

> The harmony score tells you how well the final sequence
> matches your original intent. Higher is better.

---

### Stage 9 -- ARTIFACT EXPORT

Every run produces a **case folder** with all results:

```
 data/
   peptide/
     my_case/
       sequence.fasta          <-- the biological sequence
       sequence.json           <-- full structured result
       sequence.graph.json     <-- the knowledge graph
       sequence.graph.summary  <-- graph statistics
       prompt_trace.json       <-- every decision the engine made
```

> The prompt trace is your audit log. It records every
> query variant, every score, every decision -- fully transparent.

---

## The Knowledge Graph -- What's Inside

The graph contains **8 layers** of biological knowledge:

```
 +------------------+-----------------------------------------------+
 | LAYER            | WHAT IT CONTAINS                              |
 +------------------+-----------------------------------------------+
 | Core             | Proteins, Genes, Sequence Hashes              |
 | Structure        | AlphaFold 3D, InterPro Domains, Density       |
 | Chemistry        | Minerals, Molecules, Atoms, Drug Compounds    |
 | Function         | GO Terms, Causal Models, Pathways, Evidence   |
 | Biology          | Cell Types, Components, Compartments, ncRNA   |
 | Clinical         | Annotations, Variants, Allergens, Diseases    |
 | External         | Food Sources, Microbiome, Ion Channels        |
 | Physics          | Excitation Frequencies (quantum-level)        |
 +------------------+-----------------------------------------------+
```

Everything is linked by **relationship edges** -- proteins are encoded by genes, modulate drug targets, are expressed in cell types, trigger allergen responses, and resonate at specific quantum excitation frequencies. All in one connected graph.

---

## Available Commands

| Command | What it does |
|---------|-------------|
| `generate_case_fasta` | Run the full pipeline, get a FASTA sequence back |
| `inspect_case` | Run the full pipeline, get the complete structured result |
| `generate_peptide_fasta` | Force the peptide branch |
| `generate_acid_fasta` | Force the amino acid branch |
| `solo` | Build the entire knowledge graph and return it raw |

---

## Running the Engine

```bash
# Option A: Local
pip install -r requirements.txt
python server.py                # starts on http://localhost:8000

# Option B: Docker (any OS)
docker build -t acid-master .
docker run -e GEMINI_API_KEY=your_key -p 8000:8000 acid-master

# Option C: CLI
python main.py                  # interactive prompt
```

---

## Configuration

You need one key to run:

| Variable | Required | What for |
|----------|----------|----------|
| `GEMINI_API_KEY` | yes | Powers the AI scoring, routing, harmony evaluation, and graph embedding |

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

---

## Data Sources

The engine queries **17 external scientific databases** in real time:

| Source | What it provides |
|--------|-----------------|
| UniProt | Proteins, genes, cofactors, pathways, allergens |
| ChEMBL | Drug-target pharmacology |
| PubChem | Molecular structures, atomic properties |
| ChEBI | Chemical ontology |
| Open Food Facts | Nutritional composition |
| Ensembl | Genomic coordinates, non-coding RNA |
| Reactome | Biological pathways |
| ClinPGx | Pharmacogenomic variant annotations |
| GtoPdb | Ion channel biophysics |
| VMH | Microbiome metabolic network |
| HPA | Human cell-type expression atlas |
| Cell Ontology | Standardised cell-type classification |
| AlphaFold | Predicted 3D protein structures |
| InterPro | Protein domain decomposition |
| QuickGO / GO-CAM | Gene Ontology functional annotations |
| COMPARTMENTS | Subcellular localisation |
| CTD + Open Targets | Disease associations, allergen impact |

Plus local computation: **RDKit** for molecular geometry and **PySCF** for quantum-chemical electron density matrices (DFT / TD-DFT / ddCOSMO).
