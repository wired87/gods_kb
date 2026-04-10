import asyncio
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from urllib.parse import quote

import google.generativeai as genai
import httpx
import networkx as nx
from firegraph.graph import GUtils

# ── ECO Evidence Ontology: differentiated reliability scoring ──────
# EXPERIMENTAL > HIGH_THROUGHPUT > CURATED > COMPUTATIONAL
ECO_RELIABILITY: dict[str, tuple[float, str]] = {
    # (reliability_score, evidence_type)
    "ECO:0000269": (1.0,  "EXPERIMENTAL"),   # experimental evidence – manual assertion
    "ECO:0007005": (1.0,  "EXPERIMENTAL"),   # immunofluorescence
    "ECO:0007001": (0.95, "EXPERIMENTAL"),   # immunoprecipitation
    "ECO:0000006": (0.7,  "HIGH_THROUGHPUT"),# high-throughput experimental
    "ECO:0006056": (0.7,  "HIGH_THROUGHPUT"),# high-throughput mass spec
    "ECO:0000305": (0.6,  "CURATED"),        # curator inference – manual assertion
    "ECO:0000313": (0.6,  "CURATED"),        # imported information – automatic assertion
    "ECO:0000250": (0.5,  "CURATED"),        # sequence similarity (ISS)
    "ECO:0000256": (0.3,  "COMPUTATIONAL"),  # match to sequence model
    "ECO:0000259": (0.3,  "COMPUTATIONAL"),  # match to InterPro member signature
    "ECO:0007669": (0.3,  "COMPUTATIONAL"),  # computational proteomics
    "ECO:0000501": (0.2,  "COMPUTATIONAL"),  # automatic assertion (IEA)
}
ECO_DEFAULT = (0.4, "UNKNOWN")

# ── GO short evidence codes -> ECO URIs (QuickGO returns these) ────
_GO_EV_TO_ECO: dict[str, str] = {
    "EXP": "ECO:0000269", "IDA": "ECO:0000314", "IPI": "ECO:0000353",
    "IMP": "ECO:0000315", "IGI": "ECO:0000316", "IEP": "ECO:0000270",
    "HTP": "ECO:0006056", "HDA": "ECO:0007005", "HMP": "ECO:0007001",
    "TAS": "ECO:0000304", "NAS": "ECO:0000303", "IC":  "ECO:0000305",
    "ISS": "ECO:0000250", "ISO": "ECO:0000266", "ISA": "ECO:0000247",
    "ISM": "ECO:0000255", "RCA": "ECO:0000245", "IEA": "ECO:0000501",
    "ND":  "ECO:0000307", "IBA": "ECO:0000318",
}

# ── Compound-phrase prefixes for multi-word nutrient labels ────────
COMPOUND_PHRASES = {"vitamin", "omega", "alpha", "beta", "gamma", "delta", "coenzyme", "co-enzym"}

# ── CELL TYPE INTEGRATION: memory guard ────────────────────────
_MAX_CELL_NODES = 500

# ── NON-CODING GENE DISCOVERY: memory guard ───────────────────
_MAX_NC_GENE_NODES = 1000
_NC_BIOTYPES = {"lncRNA", "miRNA", "snRNA", "snoRNA", "antisense", "lincRNA"}
_OVERLAP_FLANK_BP = 500_000  # ±500 kb around coding gene

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
}


class UniprotKB:
    def __init__(self, g):
        self.g = g
        self.client = httpx.AsyncClient(headers=_BROWSER_HEADERS, timeout=60.0)

    # --- ASYNC HELPER ---
    async def fetch_with_retry(self, url):
        while True:
            response = await self.client.get(url)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                print(f"Rate limited. Waiting {retry_after}s for {url}")
                await asyncio.sleep(retry_after)
                continue
            response.raise_for_status()
            return response.json()

    async def close(self):
        await self.client.aclose()

    # --- CORE INGESTION ---
    async def get_all_proteins(self):
        """Initialer Fetch des menschlichen Proteoms."""
        url = "https://rest.uniprot.org/uniprotkb/search?query=proteome:UP000005640"
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()

            graph_nodes = []
            graph_edges = []
            for entry in data.get("results", []):
                protein_id = entry.get("primaryAccession")
                protein_node = {
                    "id": protein_id,
                    "type": "PROTEIN",
                    "label": entry.get("uniProtkbId"),
                    "description": entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName",
                                                                                                      {}).get("value"),
                    "taxonId": entry.get("organism", {}).get("taxonId")
                }
                graph_nodes.append(protein_node)
                self._gather_genes(entry, protein_id, graph_nodes, graph_edges)
                self.apply_eco_weighting(entry, protein_id)  # Evidence direkt beim Ingest

            for node in graph_nodes: self.g.add_node(node)
            for edge in graph_edges: self.g.add_edge(**edge)
        except Exception as e:
            print(f"Error in Protein Ingestion: {e}")

    def _gather_genes(self, entry, protein_id, nodes, edges):
        for gene in entry.get("genes", []):
            gene_val = gene.get("geneName", {}).get("value")
            if not gene_val: continue
            gene_node_id = f"GENE_{gene_val}"
            nodes.append({"id": gene_node_id, "type": "GENE", "label": gene_val, "uniprot_accession": protein_id})
            edges.append({"src": protein_id, "trgt": gene_node_id,
                          "attrs": {"rel": "ENCODED_BY", "src_layer": "PROTEIN", "trgt_layer": "GENE"}})

    # --- ENRICHMENT CATEGORIES ---
    async def enrich_gene_nodes_deep(self):
        """Fetcht detaillierte Cofaktoren und Pathways für GENE nodes."""
        gene_nodes = [v for k, v in self.g.G.nodes(data=True) if v.get("type") == "GENE"]
        tasks = [self.fetch_with_retry(self.get_uniprot_url_single_gene(n["uniprot_accession"])) for n in gene_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for node, res in zip(gene_nodes, results):
            if not isinstance(res, Exception):
                node["cofactors"] = [c.get("cofactors", []) for c in res.get("comments", []) if
                                     c.get("commentType") == "COFACTOR"]
                node["reactome_pathways"] = [ref for ref in res.get("uniProtKBCrossReferences", []) if
                                             ref["database"] == "Reactome"]

    async def enrich_genomic_data(self):
        """Ensembl Integration für chromosomale Daten."""
        gene_nodes = [v for k, v in self.g.G.nodes(data=True) if v.get("type") == "GENE"]
        tasks = [self.fetch_with_retry(
            f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{n['label']}?content-type=application/json") for n in
                 gene_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for node, res in zip(gene_nodes, results):
            if not isinstance(res, Exception):
                node.update({
                    "ensembl_id": res.get("id"),
                    "chromosome": res.get("seq_region_name"),
                    "gene_start": res.get("start"),
                    "gene_end": res.get("end"),
                })

    async def enrich_molecule_chains(self):
        """Verlinkt Proteine mit Molekülketten (Reactome)."""
        protein_nodes = [v for k, v in self.g.G.nodes(data=True) if v.get("type") == "PROTEIN"]
        for node in protein_nodes:
            if "reactome_pathways" in node:
                for pw in node["reactome_pathways"]:
                    pw_id = f"MOL_{pw['id']}"
                    self.g.add_node({"id": pw_id, "type": "MOLECULE_CHAIN",
                                     "label": pw.get("properties", {}).get("pathwayName", "Pathway")})
                    self.g.add_edge(src=node["id"], trgt=pw_id, attrs={"rel": "INVOLVED_IN_CHAIN"})

    def enrich_mineral_cofactors(self):
        """Extrahiert Minerale/Metalle als Cofaktoren."""
        protein_nodes = [v for k, v in self.g.G.nodes(data=True) if v.get("type") == "PROTEIN"]
        for node in protein_nodes:
            if "cofactors" in node:
                for cluster in node["cofactors"]:
                    for cofactor in cluster:
                        m_id = f"MINERAL_{cofactor['name']}"
                        self.g.add_node({"id": m_id, "type": "MINERAL", "label": cofactor["name"]})
                        self.g.add_edge(src=node["id"], trgt=m_id, attrs={"rel": "REQUIRES_MINERAL"})

    async def enrich_functional_dynamics(self):
        """Reactome Pfade und Kausalität."""
        protein_nodes = [v for k, v in self.g.G.nodes(data=True) if v.get("type") == "PROTEIN"]
        for node in protein_nodes:
            accession = node.get("id")
            url = f"https://reactome.org/ContentService/data/pathways/low/entity/{accession}"
            try:
                res = await self.fetch_with_retry(url)
                for pw in res:
                    pw_id = f"PATHWAY_{pw['dbId']}"
                    self.g.add_node({"id": pw_id, "type": "REACTOME_PATHWAY", "label": pw['displayName']})
                    self.g.add_edge(src=accession, trgt=pw_id,
                                    attrs={"rel": "PARTICIPATES_IN", "causality": "CONTRIBUTORY"})
            except Exception:
                continue

    def apply_eco_weighting(self, entry, protein_id):
        evidences = entry.get("organism", {}).get("evidences", [])
        for ev in evidences:
            eco_code = ev.get("evidenceCode")
            if eco_code:
                reliability, evidence_type = ECO_RELIABILITY.get(eco_code, ECO_DEFAULT)
                self.g.add_node({
                    "id": eco_code, "type": "ECO_EVIDENCE", "label": eco_code,
                    "reliability": reliability,
                    "evidence_type": evidence_type,
                })
                self.g.add_edge(src=protein_id, trgt=eco_code, attrs={"rel": "VALIDATED_BY"})

    def get_uniprot_url_single_gene(self, id: str):
        fields = "accession%2Cid%2Cgene_names%2Ccc_cofactor%2Ccc_pathway%2Cxref_reactome"
        return f"https://rest.uniprot.org/uniprotkb/{id}?fields={fields}"

    # --- CATEGORY: MOLECULAR STRUCTURE (SMILES) ---
    async def _pubchem_properties_by_name(self, name: str) -> dict | None:
        """PubChem PUG REST: Name -> CanonicalSMILES + MolecularWeight + InChIKey."""
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{quote(name)}/property/CanonicalSMILES,MolecularWeight,InChIKey/JSON"
        )
        try:
            res = await self.client.get(url)
            if res.status_code == 200:
                return res.json()["PropertyTable"]["Properties"][0]
        except Exception:
            pass
        return None

    async def _chebi_fallback(self, label: str) -> dict | None:
        """ChEBI lite search as fallback when PubChem yields no hit."""
        url = f"https://www.ebi.ac.uk/chebi/advancedSearchFwd.do?searchString={quote(label)}&queryBean.stars=ALL&format=json"
        try:
            res = await self.client.get(url)
            if res.status_code == 200:
                data = res.json()
                hits = data.get("ListElement", data.get("searchResults", []))
                if hits:
                    first = hits[0] if isinstance(hits, list) else hits
                    return {"CanonicalSMILES": first.get("smiles"), "MolecularWeight": first.get("mass")}
        except Exception:
            pass
        return None

    async def enrich_molecular_structures(self):
        """
        Kategorie: MOLEKÜL -> SMILES.
        Zieht den atomaren Bauplan live von PubChem (Fallback: ChEBI).
        InChIKey dient als stabiler Primärschlüssel für Node-Deduplication.
        """
        molecule_nodes = [v for k, v in self.g.G.nodes(data=True) if v.get("type") in ["MINERAL", "MOLECULE_CHAIN"]]

        for node in molecule_nodes:
            label = node.get("label", "")

            # PRIMARY: PubChem by compound name
            props = await self._pubchem_properties_by_name(label)

            # FALLBACK: ChEBI if PubChem returned nothing
            if not props or not props.get("CanonicalSMILES"):
                props = await self._chebi_fallback(label)

            if props and props.get("CanonicalSMILES"):
                node["smiles"] = props["CanonicalSMILES"]
                node["atomic_weight"] = props.get("MolecularWeight")
                node["inchikey"] = props.get("InChIKey")
                node["smiles_source"] = "PUBCHEM" if props.get("InChIKey") else "CHEBI"
                print(f"Atomic Structure Attached: {label} (SMILES: {node['smiles']})")
            else:
                node["smiles"] = None
                node["atomic_weight"] = None
                node["smiles_source"] = "NOT_FOUND"
                print(f"SMILES not resolved for: {label}")



    @staticmethod
    def _extract_search_term(label: str) -> str:
        """Phrase-aware extraction: keeps 'Vitamin C' intact instead of splitting to 'C'."""
        tokens = label.strip().split()
        if len(tokens) >= 2 and tokens[0].lower() in COMPOUND_PHRASES:
            return " ".join(tokens[:2])
        return label

    async def enrich_food_sources(self):
        """
        Kategorie: LEBENSMITTEL -> MOLEKÜL -> PROTEIN.
        Zieht Live-Daten von Open Food Facts (OFF) für den deutschen Markt (cc=de).
        Filtert auf nutrition_grades a/b/c für Qualitätssicherung.
        """
        target_nodes = [v for k, v in self.g.G.nodes(data=True)
                        if v.get("type") in ["MINERAL", "PROTEIN"]]

        _VALID_GRADES = {"a", "b", "c"}

        for node in target_nodes:
            search_term = self._extract_search_term(node.get("label", ""))
            off_url = (
                f"https://world.openfoodfacts.org/cgi/search.pl"
                f"?search_terms={quote(search_term)}&cc=de"
                f"&search_simple=1&action=process&json=1&page_size=5"
            )

            try:
                response = await self.client.get(off_url, timeout=15.0)
                if response.status_code != 200:
                    continue
                products = response.json().get("products", [])

                for prod in products:
                    food_name = prod.get("product_name_de") or prod.get("product_name")
                    if not food_name:
                        continue

                    # QUALITY GATE: only validated nutrition grades
                    grade = (prod.get("nutrition_grades") or "").lower()
                    nutriments = prod.get("nutriments", {})
                    val = nutriments.get(f"{search_term.lower()}_100g", 0) or nutriments.get("proteins_100g", 0)

                    if val <= 0:
                        continue
                    if grade not in _VALID_GRADES and not val:
                        continue

                    food_id = f"FOOD_{prod.get('_id')}"
                    self.g.add_node({
                        "id": food_id,
                        "type": "FOOD_SOURCE",
                        "label": food_name,
                        "brand": prod.get("brands"),
                        "nova_group": prod.get("nova_group"),
                        "ecoscore": prod.get("ecoscore_grade"),
                        "nutrition_grade": grade,
                        "image_url": prod.get("image_url"),
                    })

                    self.g.add_edge(
                        src=food_id,
                        trgt=node["id"],
                        attrs={
                            "rel": "CONTAINS_NUTRIENT",
                            "amount_per_100g": val,
                            "unit": nutriments.get(f"{search_term.lower()}_unit", "g"),
                            "src_layer": "FOOD",
                            "trgt_layer": node["type"],
                        },
                    )
                    print(f"Live Linked: {food_name} [{grade}] contains {val} of {node['label']}")

            except Exception as e:
                print(f"Error fetching live food data for {search_term}: {e}")
    async def _fetch_smiles_for_drug(self, drug_name):
        """Hilfsmethode: Holt SMILES für den Wirkstoff von PubChem."""
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(drug_name)}/property/CanonicalSMILES/JSON"
        try:
            res = await self.client.get(url)
            if res.status_code == 200:
                return res.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        except Exception:
            return "N/A"

    async def enrich_pharmacology_quantum_adme(self):
        """
        Konsolidierte Inferenz-Phase: PROTEIN -> PHARMA_COMPOUND -> ATOMIC_STRUCTURE.
        1) ChEMBL mechanisms  2) ChEMBL molecule meta
        3) PubChem quantum signatures  4) Graph linking
        """
        protein_nodes = [v for k, v in self.g.G.nodes(data=True) if v.get("type") == "PROTEIN"]

        for node in protein_nodes:
            protein_id = node.get("id")
            target_label = node.get("label")

            # 1. ChEMBL: mechanisms for this UniProt target
            chembl_mech_url = f"https://www.ebi.ac.uk/chembl/api/data/mechanism?target_uniprot_accession={protein_id}&format=json"

            try:
                mech_res = await self.client.get(chembl_mech_url)
                if mech_res.status_code != 200:
                    continue
                mechanisms = mech_res.json().get("mechanisms", [])

                for mech in mechanisms:
                    mol_chembl_id = mech.get("molecule_chembl_id")

                    # 2. ChEMBL: molecule metadata (ADME & identity)
                    mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_chembl_id}?format=json"
                    mol_res = await self.client.get(mol_url)

                    if mol_res.status_code != 200:
                        continue
                    mol_data = mol_res.json()
                    struct_data = mol_data.get("molecule_structures") or {}
                    smiles_str = struct_data.get("canonical_smiles")

                    if not smiles_str:
                        continue

                    drug_name = mol_data.get("pref_name") or mol_chembl_id
                    drug_node_id = f"DRUG_{mol_chembl_id}"

                    # 3. Ingestion: Pharma-Compound Node
                    self.g.add_node({
                        "id": drug_node_id,
                        "type": "PHARMA_COMPOUND",
                        "label": drug_name,
                        "molecule_type": mol_data.get("molecule_type"),
                        "max_phase": mol_data.get("max_phase"),
                        "is_approved": mol_data.get("max_phase") == 4,
                        "chembl_id": mol_chembl_id,
                    })

                    # 4. Edge: Drug -> Protein
                    self.g.add_edge(
                        src=drug_node_id,
                        trgt=protein_id,
                        attrs={
                            "rel": "MODULATES_TARGET",
                            "action": mech.get("action_type"),
                            "mechanism": mech.get("mechanism_of_action"),
                            "src_layer": "PHARMA",
                            "trgt_layer": "PROTEIN",
                            "eco_code": "ECO:0000313",
                        },
                    )

                    # 5. PubChem: quantum-chemical & electronic signatures
                    smiles_node_id = f"SMILES_{hash(smiles_str)}"
                    pc_url = (
                        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                        f"{quote(smiles_str, safe='')}"
                        f"/property/DipoleMoment,XLogP3,Complexity,MolecularWeight,InChIKey/JSON"
                    )

                    try:
                        pc_res = await self.fetch_with_retry(pc_url)
                    except Exception:
                        pc_res = None
                    electron_props = {}
                    if pc_res and "PropertyTable" in pc_res:
                        electron_props = pc_res["PropertyTable"]["Properties"][0]

                    # 6. Ingestion: ATOMIC_STRUCTURE Node
                    self.g.add_node({
                        "id": smiles_node_id,
                        "type": "ATOMIC_STRUCTURE",
                        "label": "Molecular_SMILES",
                        "smiles": smiles_str,
                        "inchikey": electron_props.get("InChIKey"),
                        "dipole_moment": electron_props.get("DipoleMoment"),
                        "lipophilicity_logp": electron_props.get("XLogP3"),
                        "molecular_weight": electron_props.get("MolecularWeight"),
                        "complexity": electron_props.get("Complexity"),
                    })

                    # 7. Structural links
                    self.g.add_edge(
                        src=drug_node_id,
                        trgt=smiles_node_id,
                        attrs={"rel": "HAS_STRUCTURE", "src_layer": "PHARMA",
                               "trgt_layer": "ATOMIC_STRUCTURE"},
                    )
                    self.g.add_edge(
                        src=smiles_node_id,
                        trgt=protein_id,
                        attrs={"rel": "PHYSICAL_BINDING", "src_layer": "ATOMIC_STRUCTURE",
                               "trgt_layer": "PROTEIN"},
                    )

                    print(f"Enriched: {drug_name} -> Structure & Quantum Ingested")

            except Exception as e:
                print(f"Inference Error for Protein {target_label}: {e}")

    async def _fetch_smiles(self, name):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        res = await self.fetch_with_retry(url)
        return res["PropertyTable"]["Properties"][0]["CanonicalSMILES"] if res else "N/A"

    # --- SÄULE 1: PHARMAKOGENOMIK (ClinPGx / PharmGKB) ───────────────
    async def enrich_pharmacogenomics(self):
        """
        Verknüpft PHARMA_COMPOUND-Nodes mit genetischen Varianten via ClinPGx.
        Pfad: PHARMA_COMPOUND -> CLINICAL_ANNOTATION -> GENETIC_VARIANT -> GENE
        Ermöglicht personalisierte Toxizitäts- und Metabolisierungswarnungen.
        """
        _CLINPGX_BASE = "https://api.clinpgx.org/v1"
        drug_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                      if v.get("type") == "PHARMA_COMPOUND"]

        for node_id, drug in drug_nodes:
            drug_label = drug.get("label")
            if not drug_label:
                continue

            try:
                # A: ClinPGx Chemical-ID via Wirkstoffname
                chem_url = f"{_CLINPGX_BASE}/data/chemical?name={quote(drug_label)}"
                chem_res = await self.client.get(chem_url)
                if chem_res.status_code != 200:
                    continue
                chem_data = chem_res.json().get("data", [])
                if not chem_data:
                    continue

                pgkb_id = chem_data[0].get("id")
                if not pgkb_id:
                    continue

                # B: Klinische Annotationen für diesen Wirkstoff
                # ClinPGx Rate Limit: 2 req/s
                await asyncio.sleep(0.5)
                annot_url = f"{_CLINPGX_BASE}/report/connectedObjects/{pgkb_id}/ClinicalAnnotation"
                annot_res = await self.client.get(annot_url)
                if annot_res.status_code != 200:
                    continue
                annotations = annot_res.json().get("data", [])

                for annot in annotations:
                    annot_id = annot.get("id")
                    if not annot_id:
                        continue

                    # C: CLINICAL_ANNOTATION Node
                    ca_node_id = f"CLIN_ANNOT_{annot_id}"
                    self.g.add_node({
                        "id": ca_node_id,
                        "type": "CLINICAL_ANNOTATION",
                        "label": annot.get("name", annot_id),
                        "evidence_level": annot.get("evidenceLevel"),
                        "phenotype_category": annot.get("phenotypeCategory"),
                        "pgkb_id": annot_id,
                    })

                    self.g.add_edge(
                        src=node_id,
                        trgt=ca_node_id,
                        attrs={
                            "rel": "CLINICAL_SIGNIFICANCE",
                            "src_layer": "PHARMA",
                            "trgt_layer": "GENETICS",
                        },
                    )

                    # D: GENETIC_VARIANT Nodes + Rückverlinkung zu GENE
                    for var in annot.get("relatedVariants", []):
                        var_symbol = var.get("name") or var.get("symbol")
                        if not var_symbol:
                            continue

                        var_node_id = f"VARIANT_{var_symbol}"
                        self.g.add_node({
                            "id": var_node_id,
                            "type": "GENETIC_VARIANT",
                            "label": var_symbol,
                            "location": var.get("location"),
                            "pgkb_id": var.get("id"),
                        })

                        self.g.add_edge(
                            src=ca_node_id,
                            trgt=var_node_id,
                            attrs={"rel": "ASSOCIATED_VARIANT"},
                        )

                    # E: Verlinkung Variante -> bestehendes GENE via relatedGenes
                    for gene_ref in annot.get("relatedGenes", []):
                        gene_symbol = gene_ref.get("symbol") or gene_ref.get("name")
                        if not gene_symbol:
                            continue
                        target_gene_id = f"GENE_{gene_symbol}"
                        # Nur verlinken wenn der GENE-Node existiert
                        if self.g.G.has_node(target_gene_id):
                            for var in annot.get("relatedVariants", []):
                                vs = var.get("name") or var.get("symbol")
                                if vs:
                                    self.g.add_edge(
                                        src=f"VARIANT_{vs}",
                                        trgt=target_gene_id,
                                        attrs={
                                            "rel": "VARIANT_OF",
                                            "src_layer": "GENETICS",
                                            "trgt_layer": "GENE",
                                        },
                                    )

                print(f"PGx Enriched: {drug_label} ({len(annotations)} annotations)")

            except Exception as e:
                print(f"PGx Error for {drug_label}: {e}")

    # --- SÄULE 2: BIOELEKTRISCHE KNOTENEIGENSCHAFTEN (GtoPdb) ──────
    async def enrich_bioelectric_properties(self):
        """
        Integriert biophysikalische Parameter (Ionenselektivität, Leitfähigkeit,
        Spannungsabhängigkeit) für Proteine via GtoPdb.
        Pfad: PROTEIN -> ELECTRICAL_COMPONENT
        """
        _GTOP_BASE = "https://www.guidetopharmacology.org/services"
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN"]

        for node_id, protein in protein_nodes:
            uniprot_acc = protein.get("id")
            if not uniprot_acc:
                continue

            try:
                # A: GtoPdb Target-ID via UniProt Accession
                lookup_url = f"{_GTOP_BASE}/targets?accession={uniprot_acc}&database=UniProt"
                lookup_res = await self.client.get(lookup_url)
                if lookup_res.status_code != 200:
                    continue
                targets = lookup_res.json()
                if not targets:
                    continue

                target = targets[0]
                target_id = target.get("targetId")
                target_class = target.get("type", "UNKNOWN")
                if not target_id:
                    continue

                # B: Drei biophysikalische Endpunkte parallel
                sel_url = f"{_GTOP_BASE}/targets/{target_id}/ionSelectivity"
                cond_url = f"{_GTOP_BASE}/targets/{target_id}/ionConductance"
                volt_url = f"{_GTOP_BASE}/targets/{target_id}/voltageDependence"

                sel_task = self.client.get(sel_url)
                cond_task = self.client.get(cond_url)
                volt_task = self.client.get(volt_url)
                sel_res, cond_res, volt_res = await asyncio.gather(
                    sel_task, cond_task, volt_task, return_exceptions=True,
                )

                # C: Daten extrahieren (leere Responses = kein Ionenkanal)
                ion_selectivity = []
                if not isinstance(sel_res, Exception) and sel_res.status_code == 200:
                    sel_data = sel_res.json()
                    if sel_data:
                        ion_selectivity = [
                            entry.get("ion", entry.get("species", ""))
                            for entry in (sel_data if isinstance(sel_data, list) else [sel_data])
                            if entry
                        ]

                conductance_pS = None
                if not isinstance(cond_res, Exception) and cond_res.status_code == 200:
                    cond_data = cond_res.json()
                    if cond_data:
                        first_cond = cond_data[0] if isinstance(cond_data, list) else cond_data
                        conductance_pS = first_cond.get("conductance") or first_cond.get("value")

                v_half = None
                slope_factor = None
                if not isinstance(volt_res, Exception) and volt_res.status_code == 200:
                    volt_data = volt_res.json()
                    if volt_data:
                        first_volt = volt_data[0] if isinstance(volt_data, list) else volt_data
                        v_half = first_volt.get("vHalf") or first_volt.get("v_half")
                        slope_factor = first_volt.get("slopeFactor") or first_volt.get("slope")

                # D: ELECTRICAL_COMPONENT nur erstellen wenn Daten vorhanden
                if not (ion_selectivity or conductance_pS is not None or v_half is not None):
                    continue

                biophys_id = f"BIOPHYS_{uniprot_acc}"
                self.g.add_node({
                    "id": biophys_id,
                    "type": "ELECTRICAL_COMPONENT",
                    "label": f"Circuit_{target.get('name', uniprot_acc)}",
                    "target_class": target_class,
                    "ion_selectivity": ion_selectivity,
                    "conductance_pS": conductance_pS,
                    "v_half_activation": v_half,
                    "slope_factor": slope_factor,
                    "species": "Human",
                })

                self.g.add_edge(
                    src=node_id,
                    trgt=biophys_id,
                    attrs={
                        "rel": "DESCRIBED_AS_COMPONENT",
                        "src_layer": "PROTEIN",
                        "trgt_layer": "BIOELECTRIC",
                    },
                )

                print(f"Bioelectric Enriched: {protein.get('label')} -> {target_class} "
                      f"(ions={ion_selectivity}, g={conductance_pS}pS, V½={v_half})")

            except Exception as e:
                print(f"Bioelectric Error for {protein.get('label')}: {e}")

    # --- SÄULE 3: MIKROBIOM-METABOLISMUS-ACHSE (VMH) ───────────────
    async def enrich_microbiome_axis(self):
        """
        Modelliert die Transformation von Molekülen durch das Mikrobiom via VMH.
        Pfad: ATOMIC_STRUCTURE -> MICROBIAL_STRAIN -> VMH_METABOLITE -> PROTEIN
        Erklärt indirekte Wirkstoffeffekte durch bakterielle Metabolisierung.
        """
        _VMH_BASE = "https://www.vmh.life/_api"
        mol_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "ATOMIC_STRUCTURE"]

        for node_id, mol in mol_nodes:
            mol_label = mol.get("label", "")
            if not mol_label or mol_label == "Molecular_SMILES":
                # ATOMIC_STRUCTURE Nodes mit generischem Label brauchen InChIKey
                mol_label = mol.get("inchikey") or ""
            if not mol_label:
                continue

            try:
                # A: VMH Metabolit-Suche
                met_url = f"{_VMH_BASE}/metabolites/?search={quote(mol_label)}&format=json&page_size=3"
                met_res = await self.client.get(met_url, timeout=15.0)
                if met_res.status_code != 200:
                    continue
                met_data = met_res.json()
                results = met_data.get("results", [])
                if not results:
                    continue

                vmh_met = results[0]
                vmh_abbr = vmh_met.get("abbreviation")
                if not vmh_abbr:
                    continue

                # B: VMH Metabolit-Node (Brücke zwischen Graph und Mikrobiom)
                vmh_met_id = f"VMH_MET_{vmh_abbr}"
                self.g.add_node({
                    "id": vmh_met_id,
                    "type": "VMH_METABOLITE",
                    "label": vmh_met.get("fullName", vmh_abbr),
                    "vmh_abbreviation": vmh_abbr,
                    "charged_formula": vmh_met.get("chargedFormula"),
                    "inchi_string": vmh_met.get("inchiString"),
                })

                # C: Mikroben die diesen Metaboliten verarbeiten
                microbe_url = f"{_VMH_BASE}/microbes/?metabolite={quote(vmh_abbr)}&format=json&page_size=10"
                microbe_res = await self.client.get(microbe_url, timeout=15.0)
                if microbe_res.status_code != 200:
                    continue
                microbe_data = microbe_res.json().get("results", [])

                for mic in microbe_data:
                    mic_name = mic.get("organism") or mic.get("reconstruction")
                    if not mic_name:
                        continue

                    mic_node_id = f"MICROBE_{mic_name.replace(' ', '_')}"
                    self.g.add_node({
                        "id": mic_node_id,
                        "type": "MICROBIAL_STRAIN",
                        "label": mic_name,
                        "phylum": mic.get("phylum"),
                        "family": mic.get("family"),
                        "metabolic_role": "CONSUMER",
                    })

                    # Edge: Quellmolekül -> Mikrobe
                    self.g.add_edge(
                        src=node_id,
                        trgt=mic_node_id,
                        attrs={
                            "rel": "METABOLIZED_BY",
                            "src_layer": "ATOMIC",
                            "trgt_layer": "MICROBIOME",
                        },
                    )

                # D: Fermentationsprodukte dieses Metaboliten
                ferm_url = f"{_VMH_BASE}/fermcarbon/?metabolite={quote(vmh_abbr)}&format=json&page_size=10"
                ferm_res = await self.client.get(ferm_url, timeout=15.0)
                if ferm_res.status_code == 200:
                    ferm_data = ferm_res.json().get("results", [])
                    for ferm in ferm_data:
                        ferm_model = ferm.get("model")
                        source_type = ferm.get("sourcetype", "")
                        if not ferm_model:
                            continue

                        # Fermentations-Mikrobe als PRODUCER markieren
                        ferm_mic_id = f"MICROBE_{ferm_model.replace(' ', '_')}"
                        self.g.add_node({
                            "id": ferm_mic_id,
                            "type": "MICROBIAL_STRAIN",
                            "label": ferm_model,
                            "metabolic_role": "PRODUCER" if "Fermentation" in source_type else "CONSUMER",
                        })

                        self.g.add_edge(
                            src=ferm_mic_id,
                            trgt=vmh_met_id,
                            attrs={
                                "rel": "PRODUCES_METABOLITE",
                                "source_type": source_type,
                                "src_layer": "MICROBIOME",
                                "trgt_layer": "METABOLITE",
                            },
                        )

                # E: Rückverlinkung VMH_METABOLITE -> PROTEIN via VMH Gene-Mapping
                gene_url = f"{_VMH_BASE}/genes/?reaction={quote(vmh_abbr)}&format=json&page_size=5"
                gene_res = await self.client.get(gene_url, timeout=15.0)
                if gene_res.status_code == 200:
                    gene_hits = gene_res.json().get("results", [])
                    for gh in gene_hits:
                        gene_symbol = gh.get("symbol")
                        if not gene_symbol:
                            continue
                        # Suche passendes PROTEIN über GENE-Node
                        gene_node_id = f"GENE_{gene_symbol}"
                        if self.g.G.has_node(gene_node_id):
                            gene_attrs = self.g.G.nodes[gene_node_id]
                            protein_acc = gene_attrs.get("uniprot_accession")
                            if protein_acc and self.g.G.has_node(protein_acc):
                                self.g.add_edge(
                                    src=vmh_met_id,
                                    trgt=protein_acc,
                                    attrs={
                                        "rel": "TARGETS_HUMAN",
                                        "src_layer": "METABOLITE",
                                        "trgt_layer": "PROTEIN",
                                    },
                                )

                print(f"Microbiome Enriched: {mol_label} -> VMH:{vmh_abbr} "
                      f"({len(microbe_data)} microbes)")

            except Exception as e:
                print(f"Microbiome Error for {mol_label}: {e}")

    # --- SÄULE 4: ZELLULÄRE INTEGRATION (HPA + Cell Ontology) ──────

    @staticmethod
    def _parse_hpa_cell_enrichment(rtcte_value: str) -> list[str]:
        """
        HPA 'RNA tissue cell type enrichment' -> list of cell-type labels.
        Format: 'Cell type enriched (Hepatocytes);Group enriched (Epithelial cells)'
        """
        if not rtcte_value:
            return []
        _SKIP = {"not detected", "low cell type specificity", "n/a", ""}
        cell_types: list[str] = []
        for segment in rtcte_value.split(";"):
            segment = segment.strip()
            if segment.lower() in _SKIP:
                continue
            paren_start = segment.find("(")
            paren_end = segment.rfind(")")
            if paren_start != -1 and paren_end > paren_start:
                inner = segment[paren_start + 1:paren_end].strip()
                for ct in inner.split(","):
                    ct = ct.strip()
                    if ct:
                        cell_types.append(ct)
        return cell_types

    async def _resolve_cell_ontology(self, cell_label: str) -> dict | None:
        """
        OLS4 CL lookup: cell_label -> CL-ID + description + parent + marker genes.
        Exact match first, fuzzy fallback for plurals/synonyms.
        """
        _OLS4_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
        try:
            # EXACT match against Cell Ontology
            res = await self.client.get(
                _OLS4_SEARCH,
                params={"q": cell_label, "ontology": "cl", "exact": "true", "rows": 1},
                timeout=15.0,
            )
            if res.status_code != 200:
                return None
            docs = res.json().get("response", {}).get("docs", [])

            # FALLBACK: fuzzy search for plural forms / synonyms
            if not docs:
                res = await self.client.get(
                    _OLS4_SEARCH,
                    params={"q": cell_label, "ontology": "cl", "rows": 1},
                    timeout=15.0,
                )
                if res.status_code != 200:
                    return None
                docs = res.json().get("response", {}).get("docs", [])
                if not docs:
                    return None

            hit = docs[0]
            obo_id = hit.get("obo_id", "")
            cl_id = obo_id.replace(":", "_") if obo_id else None
            if not cl_id or not cl_id.startswith("CL_"):
                return None

            desc_raw = hit.get("description", [])
            description = ". ".join(desc_raw) if isinstance(desc_raw, list) else str(desc_raw or "")

            # MARKER GENES from CL annotation (if available in this release)
            marker_genes = hit.get("annotation", {}).get("has_marker_gene", [])

            return {
                "cl_id": cl_id,
                "label": hit.get("label", cell_label),
                "description": description,
                "marker_genes": marker_genes,
            }
        except Exception as e:
            print(f"OLS4 Error for '{cell_label}': {e}")
            return None

    async def enrich_cell_type_expression(self):
        """
        Zelluläre Integration: GENE -> CELL_TYPE via HPA Expression + OLS4 Metadaten.
        Two-Pass:
          1) HPA rtcte -> provisorische CELL_TYPE Nodes + EXPRESSED_IN_CELL Kanten
          2) OLS4 CL  -> description, CL-ID, parent + HAS_MARKER_GENE Rückverkettung
        """
        _seen: set[str] = set()
        _count = 0

        gene_nodes = [(k, v) for k, v in self.g.G.nodes(data=True) if v.get("type") == "GENE"]

        # ── PASS 1: HPA Expression Data ──────────────────────────────
        for gene_id, gene in gene_nodes:
            if _count >= _MAX_CELL_NODES:
                print(f"CELL_TYPE cap reached ({_MAX_CELL_NODES})")
                break

            gene_name = gene.get("label")
            if not gene_name:
                continue

            hpa_url = (
                f"https://www.proteinatlas.org/api/search_download.php"
                f"?search={quote(gene_name)}&format=json"
                f"&columns=g,eg,up,rtcte&compress=no"
            )
            try:
                res = await self.client.get(hpa_url, timeout=20.0)
                if res.status_code != 200:
                    continue
                entries = res.json()
                if not isinstance(entries, list) or not entries:
                    continue

                # HPA may return multiple genes; prefer exact name match
                matched = next(
                    (e for e in entries if (e.get("Gene") or "").upper() == gene_name.upper()),
                    entries[0],
                )
                rtcte = matched.get("RNA tissue cell type enrichment", "")
                cell_types = self._parse_hpa_cell_enrichment(rtcte)

                for ct_label in cell_types:
                    ct_key = ct_label.lower().strip()
                    cell_node_id = f"CELL_{ct_key.replace(' ', '_').upper()}"

                    if ct_key not in _seen and _count < _MAX_CELL_NODES:
                        self.g.add_node({
                            "id": cell_node_id,
                            "type": "CELL_TYPE",
                            "label": ct_label,
                            "ontology_prefix": "CL",
                            "cl_resolved": False,
                        })
                        _seen.add(ct_key)
                        _count += 1

                    if self.g.G.has_node(cell_node_id):
                        self.g.add_edge(
                            src=gene_id, trgt=cell_node_id,
                            attrs={"rel": "EXPRESSED_IN_CELL", "src_layer": "GENE", "trgt_layer": "CELL"},
                        )

            except Exception as e:
                print(f"HPA Error for {gene_name}: {e}")

        print(f"Pass 1 done: {_count} CELL_TYPE nodes from HPA expression data")

        # ── PASS 2: OLS4 Cell Ontology Metadata ─────────────────────
        unresolved = [(k, v) for k, v in self.g.G.nodes(data=True)
                      if v.get("type") == "CELL_TYPE" and not v.get("cl_resolved")]

        resolved = 0
        for cell_id, cell in unresolved:
            cl_data = await self._resolve_cell_ontology(cell.get("label", ""))
            if not cl_data:
                continue

            cell["cl_id"] = cl_data["cl_id"]
            cell["label"] = cl_data["label"]
            cell["description"] = cl_data["description"]
            cell["cl_resolved"] = True
            resolved += 1

            # HAS_MARKER_GENE: CL annotation -> existing GENE nodes
            for marker in cl_data.get("marker_genes", []):
                target_gene = f"GENE_{marker}"
                if self.g.G.has_node(target_gene):
                    self.g.add_edge(
                        src=cell_id, trgt=target_gene,
                        attrs={"rel": "HAS_MARKER_GENE", "src_layer": "CELL", "trgt_layer": "GENE"},
                    )

        print(f"Pass 2 done: {resolved}/{_count} cells resolved via Cell Ontology (OLS4)")

    # ═══════════════════════════════════════════════════════════════════
    # LAYER: SEQUENCE IDENTITY HASHING (SHA-256 Content Addressing)
    # Dedupliziert identische Sequenzen aus verschiedenen Quellen und
    # ermöglicht Fragment-Matching gegen bekannte toxische/allergene Motive.
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _sha256_sequence(seq: str) -> str:
        """Normalisierter SHA-256: uppercase, whitespace-stripped."""
        return hashlib.sha256(seq.strip().upper().encode("utf-8")).hexdigest()

    def compute_sequence_hashes(self):
        """
        Content-Addressing: Jede Aminosäuresequenz bekommt einen SHA-256 Hash-Node.
        Proteine mit identischer Sequenz werden über SEQUENCE_HASH verschmolzen.
        """
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN"]
        _hash_map: dict[str, list[str]] = {}
        hashed = 0

        for node_id, protein in protein_nodes:
            seq = protein.get("sequence")
            if not seq:
                continue

            seq_hash = self._sha256_sequence(seq)
            protein["sequence_hash"] = seq_hash

            if seq_hash not in _hash_map:
                _hash_map[seq_hash] = []
            _hash_map[seq_hash].append(node_id)

        for seq_hash, protein_ids in _hash_map.items():
            hash_node_id = f"SEQHASH_{seq_hash[:16]}"
            self.g.add_node({
                "id": hash_node_id,
                "type": "SEQUENCE_HASH",
                "label": f"SHA256:{seq_hash[:16]}",
                "full_hash": seq_hash,
                "sequence_count": len(protein_ids),
            })
            for pid in protein_ids:
                self.g.add_edge(
                    src=pid, trgt=hash_node_id,
                    attrs={
                        "rel": "HAS_SEQUENCE_IDENTITY",
                        "src_layer": "PROTEIN",
                        "trgt_layer": "SEQUENCE_HASH",
                    },
                )
            hashed += len(protein_ids)

        # DEDUP MARKER: wenn mehrere Proteine denselben Hash teilen
        shared = {h: ids for h, ids in _hash_map.items() if len(ids) > 1}
        for seq_hash, protein_ids in shared.items():
            hash_node_id = f"SEQHASH_{seq_hash[:16]}"
            for i, pid_a in enumerate(protein_ids):
                for pid_b in protein_ids[i + 1:]:
                    self.g.add_edge(
                        src=pid_a, trgt=pid_b,
                        attrs={
                            "rel": "SEQUENCE_IDENTICAL",
                            "via_hash": hash_node_id,
                            "src_layer": "PROTEIN",
                            "trgt_layer": "PROTEIN",
                        },
                    )

        print(f"Sequence Hashing: {hashed} proteins hashed, "
              f"{len(shared)} shared sequences detected")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 11: STRUCTURAL INFERENCE (AlphaFold DB)
    # Verknüpft Proteine mit vorhergesagten 3D-Strukturen.
    # pLDDT-Score dient als Qualitätsmetrik der Vorhersage.
    # ═══════════════════════════════════════════════════════════════════

    async def enrich_structural_layer(self):
        """
        AlphaFold DB Integration: PROTEIN -[HAS_PREDICTED_STRUCTURE]-> 3D_STRUCTURE.
        pLDDT-Score als Zuverlässigkeitsmetrik an der Kante.
        """
        _AF_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN"]
        linked = 0

        for node_id, protein in protein_nodes:
            accession = protein.get("id")
            if not accession:
                continue

            try:
                res = await self.client.get(f"{_AF_BASE}/{accession}", timeout=20.0)
                if res.status_code != 200:
                    continue
                entries = res.json()
                if not entries:
                    continue
                data = entries[0] if isinstance(entries, list) else entries

                model_id = f"STRUCT_{accession}"
                plddt = data.get("globalMetrics", {}).get("globalPlddt") or data.get("uniprotScore")

                self.g.add_node({
                    "id": model_id,
                    "type": "3D_STRUCTURE",
                    "label": f"AlphaFold_{accession}",
                    "pLDDT_avg": plddt,
                    "pdb_url": data.get("pdbUrl"),
                    "cif_url": data.get("cifUrl"),
                    "model_version": data.get("latestVersion"),
                    "gene": data.get("gene"),
                })

                self.g.add_edge(
                    src=node_id, trgt=model_id,
                    attrs={
                        "rel": "HAS_PREDICTED_STRUCTURE",
                        "pLDDT": plddt,
                        "src_layer": "PROTEIN",
                        "trgt_layer": "STRUCTURAL",
                    },
                )
                linked += 1

            except Exception as e:
                print(f"AlphaFold Error for {accession}: {e}")

        print(f"Phase 11: {linked} proteins linked to AlphaFold structures")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 12: DOMAIN DECOMPOSITION (InterPro)
    # Bricht Proteine in funktionale Domänen auf.
    # Ermöglicht funktionale Ähnlichkeitsschlüsse zwischen neuartigen
    # und bekannten Proteinen über gemeinsame Domänen.
    # ═══════════════════════════════════════════════════════════════════

    async def enrich_domain_decomposition(self):
        """
        InterPro Integration: PROTEIN -[CONTAINS_DOMAIN]-> DOMAIN.
        Domänen aus Pfam, SMART, CDD etc. werden als eigenständige Nodes modelliert.
        """
        _INTERPRO_BASE = "https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot"
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN"]
        domain_count = 0

        for node_id, protein in protein_nodes:
            accession = protein.get("id")
            if not accession:
                continue

            try:
                res = await self.client.get(
                    f"{_INTERPRO_BASE}/{accession}",
                    headers={"Accept": "application/json"},
                    timeout=20.0,
                )
                if res.status_code != 200:
                    continue
                payload = res.json()
                results = payload.get("results", [])

                for entry in results:
                    meta = entry.get("metadata", {})
                    ipr_id = meta.get("accession")
                    if not ipr_id:
                        continue

                    domain_node_id = f"DOMAIN_{ipr_id}"
                    # IDEMPOTENT: Domain-Node nur einmal anlegen, mehrfach verlinken
                    if not self.g.G.has_node(domain_node_id):
                        self.g.add_node({
                            "id": domain_node_id,
                            "type": "PROTEIN_DOMAIN",
                            "label": meta.get("name", ipr_id),
                            "interpro_id": ipr_id,
                            "domain_type": meta.get("type"),
                            "source_database": meta.get("source_database"),
                        })

                        # EXPAND: InterPro GO terms -> proper GO_TERM nodes + edges
                        for go_raw in meta.get("go_terms", []):
                            go_ident = go_raw.get("identifier")
                            if not go_ident:
                                continue
                            go_node_id = f"GO_{go_ident.replace(':', '_')}"
                            if not self.g.G.has_node(go_node_id):
                                self.g.add_node({
                                    "id": go_node_id,
                                    "type": "GO_TERM",
                                    "label": go_raw.get("name", go_ident),
                                    "go_id": go_ident,
                                    "aspect": go_raw.get("category", {}).get("name"),
                                })
                            self.g.add_edge(
                                src=domain_node_id, trgt=go_node_id,
                                attrs={
                                    "rel": "ASSOCIATED_GO",
                                    "src_layer": "DOMAIN",
                                    "trgt_layer": "FUNCTIONAL",
                                },
                            )

                    # POSITIONALE INFORMATION: wo sitzt die Domäne in der Sequenz?
                    proteins_block = entry.get("proteins", [])
                    locations = []
                    for prot_entry in proteins_block:
                        for loc_group in prot_entry.get("entry_protein_locations", []):
                            for frag in loc_group.get("fragments", []):
                                locations.append({
                                    "start": frag.get("start"),
                                    "end": frag.get("end"),
                                })

                    self.g.add_edge(
                        src=node_id, trgt=domain_node_id,
                        attrs={
                            "rel": "CONTAINS_DOMAIN",
                            "positions": locations if locations else None,
                            "src_layer": "PROTEIN",
                            "trgt_layer": "DOMAIN",
                        },
                    )
                    domain_count += 1

            except Exception as e:
                print(f"InterPro Error for {accession}: {e}")

        print(f"Phase 12: {domain_count} domain links created")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 13: GO-SEMANTIC-LINKING (Gene Ontology via QuickGO)
    # + COMPARTMENTS LOCALIZATION
    # Erstellt ein semantisches Netz aus Funktionen (Molecular Function,
    # Biological Process) und subzellulärer Lokalisation.
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_go_evidence(raw_code: str) -> tuple[float, str, str]:
        """Resolve QuickGO evidence (short GO code OR ECO URI) -> (reliability, evidence_type, eco_uri)."""
        eco_uri = _GO_EV_TO_ECO.get(raw_code, raw_code)
        reliability, evidence_type = ECO_RELIABILITY.get(eco_uri, ECO_DEFAULT)
        return reliability, evidence_type, eco_uri

    async def enrich_go_semantic_layer(self):
        """
        Gene Ontology via QuickGO: PROTEIN -[ANNOTATED_WITH]-> GO_TERM.
        Evidence codes resolved through short GO codes AND ECO URIs.
        Only annotations with reliability >= 0.5 are kept.
        """
        _QUICKGO_BASE = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
        _MIN_RELIABILITY = 0.5
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN"]
        go_count = 0

        for node_id, protein in protein_nodes:
            accession = protein.get("id")
            if not accession:
                continue

            try:
                res = await self.client.get(
                    _QUICKGO_BASE,
                    params={"geneProductId": accession, "limit": 100, "taxonId": 9606},
                    headers={"Accept": "application/json"},
                    timeout=20.0,
                )
                if res.status_code != 200:
                    continue
                results = res.json().get("results", [])

                for anno in results:
                    go_id = anno.get("goId")
                    if not go_id:
                        continue

                    # RESOLVE: short GO evidence codes (IDA, IPI, …) + full ECO URIs
                    raw_code = anno.get("goEvidence", "")
                    reliability, evidence_type, eco_uri = self._resolve_go_evidence(raw_code)
                    if reliability < _MIN_RELIABILITY:
                        continue

                    go_node_id = f"GO_{go_id.replace(':', '_')}"
                    if not self.g.G.has_node(go_node_id):
                        self.g.add_node({
                            "id": go_node_id,
                            "type": "GO_TERM",
                            "label": anno.get("goName", go_id),
                            "go_id": go_id,
                            "aspect": anno.get("goAspect"),
                        })

                    self.g.add_edge(
                        src=node_id, trgt=go_node_id,
                        attrs={
                            "rel": "ANNOTATED_WITH",
                            "evidence_code": eco_uri,
                            "reliability": reliability,
                            "evidence_type": evidence_type,
                            "qualifier": anno.get("qualifier"),
                            "assigned_by": anno.get("assignedBy"),
                            "extension": anno.get("extensions"),
                            "src_layer": "PROTEIN",
                            "trgt_layer": "FUNCTIONAL",
                        },
                    )
                    go_count += 1

            except Exception as e:
                print(f"QuickGO Error for {accession}: {e}")

        print(f"Phase 13a: {go_count} GO annotations linked (min reliability={_MIN_RELIABILITY})")

    # ── GO TERM METADATA ENRICHMENT ──────────────────────────────────
    async def _enrich_go_term_metadata(self):
        """
        Batch-fetch GO term definitions, synonyms, obsolescence, comments
        from QuickGO /ontology/go/terms/{ids} (up to 25 IDs per call).
        Patches existing GO_TERM nodes in-place.
        """
        _TERM_BASE = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"
        _BATCH_SIZE = 25
        go_nodes = [(n, d) for n, d in self.g.G.nodes(data=True) if d.get("type") == "GO_TERM"]
        enriched = 0

        for i in range(0, len(go_nodes), _BATCH_SIZE):
            chunk = go_nodes[i : i + _BATCH_SIZE]
            ids_csv = ",".join(d["go_id"] for _, d in chunk if d.get("go_id"))
            if not ids_csv:
                continue

            try:
                res = await self.client.get(
                    f"{_TERM_BASE}/{quote(ids_csv, safe='')}",
                    headers={"Accept": "application/json"},
                    timeout=25.0,
                )
                if res.status_code != 200:
                    continue

                for term in res.json().get("results", []):
                    tid = term.get("id")
                    if not tid:
                        continue
                    node_id = f"GO_{tid.replace(':', '_')}"
                    if not self.g.G.has_node(node_id):
                        continue

                    self.g.G.nodes[node_id].update({
                        "definition": (term.get("definition") or {}).get("text"),
                        "synonyms": [s.get("name") for s in term.get("synonyms", []) if s.get("name")],
                        "is_obsolete": term.get("isObsolete", False),
                        "comment": term.get("comment"),
                    })
                    enriched += 1

            except Exception as e:
                print(f"GO Term Metadata Error: {e}")

        print(f"Phase 13a+: {enriched}/{len(go_nodes)} GO_TERM nodes enriched with metadata")

    # ── GO ONTOLOGY HIERARCHY ────────────────────────────────────────
    async def _wire_go_hierarchy(self):
        """
        Build IS_A / PART_OF / REGULATES edges between GO_TERM nodes
        already present in the graph. Uses QuickGO /ontology/go/terms/{ids}/children.
        Only creates edges when BOTH endpoints exist in the graph.
        """
        _CHILDREN_BASE = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"
        _BATCH_SIZE = 25
        _VALID_RELS = {"is_a", "part_of", "regulates", "positively_regulates", "negatively_regulates"}
        go_nodes = [(n, d) for n, d in self.g.G.nodes(data=True) if d.get("type") == "GO_TERM"]
        # SET of all GO node IDs for fast membership check
        go_ids_in_graph = {n for n, _ in go_nodes}
        hierarchy_count = 0

        for i in range(0, len(go_nodes), _BATCH_SIZE):
            chunk = go_nodes[i : i + _BATCH_SIZE]
            ids_csv = ",".join(d["go_id"] for _, d in chunk if d.get("go_id"))
            if not ids_csv:
                continue

            try:
                res = await self.client.get(
                    f"{_CHILDREN_BASE}/{quote(ids_csv, safe='')}/children",
                    headers={"Accept": "application/json"},
                    timeout=25.0,
                )
                if res.status_code != 200:
                    continue

                for term in res.json().get("results", []):
                    parent_go = term.get("id")
                    if not parent_go:
                        continue
                    parent_node_id = f"GO_{parent_go.replace(':', '_')}"
                    if parent_node_id not in go_ids_in_graph:
                        continue

                    for child in term.get("children", []):
                        child_go = child.get("id")
                        relation = child.get("relation", "").lower()
                        if not child_go or relation not in _VALID_RELS:
                            continue

                        child_node_id = f"GO_{child_go.replace(':', '_')}"
                        if child_node_id not in go_ids_in_graph:
                            continue

                        # DIRECTION: child -[IS_A]-> parent (standard GO convention)
                        self.g.add_edge(
                            src=child_node_id, trgt=parent_node_id,
                            attrs={
                                "rel": relation.upper(),
                                "src_layer": "GO_HIERARCHY",
                                "trgt_layer": "GO_HIERARCHY",
                            },
                        )
                        hierarchy_count += 1

            except Exception as e:
                print(f"GO Hierarchy Error: {e}")

        print(f"Phase 13a++: {hierarchy_count} GO hierarchy edges created")

    # ── GENE -> GO_TERM DERIVED EDGES ────────────────────────────────
    _ASPECT_TO_REL = {
        "molecular_function": "HAS_FUNCTION",
        "biological_process": "INVOLVED_IN_PROCESS",
        "cellular_component": "LOCATED_IN_COMPONENT",
    }

    def _wire_gene_go_edges(self):
        """
        Derive GENE -> GO_TERM edges by traversing
        PROTEIN -[ANNOTATED_WITH]-> GO_TERM and PROTEIN -[ENCODED_BY]-> GENE.
        Uses the strongest reliability per (gene, go_term) pair.
        """
        # COLLECT: protein -> gene mapping
        protein_to_genes: dict[str, list[str]] = {}
        for src, trgt, edata in self.g.G.edges(data=True):
            if edata.get("rel") == "ENCODED_BY":
                protein_to_genes.setdefault(src, []).append(trgt)

        # COLLECT: protein -> go_term annotations with best reliability
        # KEY: (gene_node_id, go_node_id) -> best reliability
        gene_go_best: dict[tuple[str, str], tuple[float, str, str]] = {}

        for src, trgt, edata in self.g.G.edges(data=True):
            if edata.get("rel") != "ANNOTATED_WITH":
                continue
            protein_id = src
            go_node_id = trgt
            reliability = edata.get("reliability", 0)
            evidence_type = edata.get("evidence_type", "")
            go_attrs = self.g.G.nodes.get(go_node_id, {})
            aspect = go_attrs.get("aspect", "")

            for gene_id in protein_to_genes.get(protein_id, []):
                key = (gene_id, go_node_id)
                if key not in gene_go_best or reliability > gene_go_best[key][0]:
                    gene_go_best[key] = (reliability, evidence_type, protein_id)

        derived_count = 0
        for (gene_id, go_node_id), (reliability, evidence_type, protein_id) in gene_go_best.items():
            go_attrs = self.g.G.nodes.get(go_node_id, {})
            aspect = go_attrs.get("aspect", "")
            rel = self._ASPECT_TO_REL.get(aspect, "ANNOTATED_WITH_GENE")

            self.g.add_edge(
                src=gene_id, trgt=go_node_id,
                attrs={
                    "rel": rel,
                    "derived_from": protein_id,
                    "reliability": reliability,
                    "evidence_type": evidence_type,
                    "src_layer": "GENE",
                    "trgt_layer": "FUNCTIONAL",
                },
            )
            derived_count += 1

        print(f"Phase 13a+++: {derived_count} GENE -> GO_TERM derived edges created")

    async def enrich_compartment_localization(self):
        """
        COMPARTMENTS DB (JensenLab): PROTEIN -[LOCALIZED_IN]-> COMPARTMENT.
        Subzelluläre Lokalisation mit Konfidenz-Score.
        """
        _COMP_BASE = "https://compartments.jensenlab.org/Entity"
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN"]
        loc_count = 0
        _MIN_CONFIDENCE = 2.0

        for node_id, protein in protein_nodes:
            accession = protein.get("id")
            gene_label = protein.get("label", "")
            if not accession:
                continue

            try:
                res = await self.client.get(
                    _COMP_BASE,
                    params={"query": gene_label, "type": "9606", "format": "json"},
                    timeout=15.0,
                )
                if res.status_code != 200:
                    continue
                entries = res.json()
                if not isinstance(entries, list):
                    continue

                for entry in entries:
                    compartment = entry.get("compartment", {})
                    comp_id_raw = compartment.get("id") or entry.get("go_id")
                    comp_name = compartment.get("name") or entry.get("name")
                    confidence = float(entry.get("confidence", 0))

                    if not comp_id_raw or not comp_name:
                        continue
                    if confidence < _MIN_CONFIDENCE:
                        continue

                    comp_node_id = f"COMP_{comp_id_raw.replace(':', '_')}"
                    if not self.g.G.has_node(comp_node_id):
                        self.g.add_node({
                            "id": comp_node_id,
                            "type": "COMPARTMENT",
                            "label": comp_name,
                            "go_id": comp_id_raw,
                        })

                        # CROSS-LINK: if go_id matches an existing GO_TERM node, wire them
                        if comp_id_raw and comp_id_raw.startswith("GO:"):
                            mapped_go_node = f"GO_{comp_id_raw.replace(':', '_')}"
                            if self.g.G.has_node(mapped_go_node):
                                self.g.add_edge(
                                    src=comp_node_id, trgt=mapped_go_node,
                                    attrs={
                                        "rel": "MAPPED_TO_GO",
                                        "src_layer": "LOCALIZATION",
                                        "trgt_layer": "FUNCTIONAL",
                                    },
                                )

                    self.g.add_edge(
                        src=node_id, trgt=comp_node_id,
                        attrs={
                            "rel": "LOCALIZED_IN",
                            "confidence": confidence,
                            "source": entry.get("source"),
                            "src_layer": "PROTEIN",
                            "trgt_layer": "LOCALIZATION",
                        },
                    )
                    loc_count += 1

            except Exception as e:
                print(f"COMPARTMENTS Error for {gene_label}: {e}")

        print(f"Phase 13b: {loc_count} localization links created (min confidence={_MIN_CONFIDENCE})")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 13c: GO-CAM CAUSAL ACTIVITY MODELS (BioLink API)
    # Fetches gene-function associations and builds GOCAM_ACTIVITY nodes
    # with causal/structural edges into the GO semantic layer.
    # API: http://api.geneontology.org/api/bioentity/gene/UniProtKB:{acc}/function
    # ═══════════════════════════════════════════════════════════════════

    _BIOLINK_BASE = "http://api.geneontology.org/api"
    _GOCAM_ROWS = 50

    async def enrich_gocam_activities(self):
        """
        GO-CAM via BioLink API: GOCAM_ACTIVITY nodes + causal edges.
        For each GENE node with a uniprot_accession, queries functional
        associations and wires them through self.g (GUtils).
        """
        gene_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                      if v.get("type") == "GENE"]
        activity_count = 0
        edge_count = 0
        _seen: set[str] = set()

        for gene_id, gene_data in gene_nodes:
            accession = gene_data.get("uniprot_accession")
            if not accession:
                continue

            url = f"{self._BIOLINK_BASE}/bioentity/gene/UniProtKB:{accession}/function"
            try:
                res = await self.client.get(
                    url,
                    params={"rows": self._GOCAM_ROWS},
                    headers={"Accept": "application/json"},
                    timeout=25.0,
                )
                if res.status_code != 200:
                    continue

                for assoc in res.json().get("associations", []):
                    obj = assoc.get("object", {})
                    go_id = obj.get("id")
                    go_label = obj.get("label", "")
                    go_category = obj.get("category", [])
                    if not go_id:
                        continue

                    rel_label = (assoc.get("relation") or {}).get("label", "associated_with")

                    # UNIQUE: one activity per (accession, GO term, relation)
                    act_id = f"GOCAM_{accession}_{go_id.replace(':', '_')}_{rel_label}"
                    if act_id in _seen:
                        continue
                    _seen.add(act_id)

                    # EVIDENCE + PROVENANCE
                    evidence_types = [
                        ev.get("label") for ev in assoc.get("evidence_types", [])
                        if ev.get("label")
                    ]
                    provided_by = [
                        s for s in assoc.get("provided_by", [])
                        if isinstance(s, str)
                    ]

                    # ENSURE: GO_TERM node exists
                    go_node_id = f"GO_{go_id.replace(':', '_')}"
                    if not self.g.G.has_node(go_node_id):
                        self.g.add_node({
                            "id": go_node_id,
                            "type": "GO_TERM",
                            "label": go_label,
                            "go_id": go_id,
                            "aspect": self._infer_go_aspect(go_category),
                        })

                    # CREATE: GOCAM_ACTIVITY node
                    self.g.add_node({
                        "id": act_id,
                        "type": "GOCAM_ACTIVITY",
                        "label": f"{gene_data.get('label', '')} {rel_label} {go_label}",
                        "activity_relation": rel_label,
                        "evidence_types": evidence_types,
                        "provided_by": provided_by,
                    })
                    activity_count += 1

                    # EDGE: GOCAM_ACTIVITY -[ENABLED_BY]-> GENE
                    self.g.add_edge(
                        src=act_id, trgt=gene_id,
                        attrs={"rel": "ENABLED_BY", "src_layer": "GOCAM", "trgt_layer": "GENE"},
                    )
                    edge_count += 1

                    # EDGE: GOCAM_ACTIVITY -[aspect-rel]-> GO_TERM
                    go_rel = self._gocam_edge_rel(rel_label, go_category)
                    self.g.add_edge(
                        src=act_id, trgt=go_node_id,
                        attrs={"rel": go_rel, "src_layer": "GOCAM", "trgt_layer": "FUNCTIONAL"},
                    )
                    edge_count += 1

            except Exception as e:
                print(f"GO-CAM Error for {accession}: {e}")

        print(f"Phase 13c: {activity_count} GOCAM_ACTIVITY nodes, {edge_count} causal edges created")

    @staticmethod
    def _infer_go_aspect(categories: list) -> str | None:
        """Map BioLink category list to GO aspect string."""
        for cat in categories:
            low = cat.lower() if isinstance(cat, str) else ""
            if "molecular" in low or "activity" in low:
                return "molecular_function"
            if "biological" in low or "process" in low:
                return "biological_process"
            if "cellular" in low or "component" in low:
                return "cellular_component"
        return None

    @staticmethod
    def _gocam_edge_rel(rel_label: str, categories: list) -> str:
        """Choose GOCAM edge rel based on GO aspect and association relation."""
        for cat in categories:
            low = cat.lower() if isinstance(cat, str) else ""
            if "biological" in low or "process" in low:
                return "PART_OF"
            if "cellular" in low or "component" in low:
                return "OCCURS_IN"
        if "enables" in rel_label.lower():
            return "ENABLES"
        if "contributes" in rel_label.lower():
            return "CONTRIBUTES_TO"
        return "ASSOCIATED_WITH"

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 14: ALLERGEN DETECTION (UniProt KW-0020 + Description Scan)
    # Identifiziert allergene Proteine im Graph und erstellt ALLERGEN-
    # Nodes als Brücke zu immunologischen Pfaden.
    # ═══════════════════════════════════════════════════════════════════

    async def detect_allergen_proteins(self):
        """
        Scannt PROTEIN-Nodes auf allergenes Potenzial.
        Primär: UniProt Keyword KW-0020 ("Allergen") via API.
        Fallback: Description-basierte Detektion für bereits geladene Proteine.
        """
        # A: UniProt-Suche nach annotierten humanen Allergenen (KW-0020)
        url = (
            "https://rest.uniprot.org/uniprotkb/search"
            "?query=keyword:KW-0020+AND+organism_id:9606"
            "&fields=accession,id,protein_name,gene_names,keyword"
            "&format=json&size=500"
        )
        allergen_accessions: set[str] = set()
        try:
            res = await self.client.get(url, timeout=30.0)
            if res.status_code == 200:
                for entry in res.json().get("results", []):
                    acc = entry.get("primaryAccession")
                    if acc:
                        allergen_accessions.add(acc)
        except Exception as e:
            print(f"UniProt Allergen KW-0020 Query Error: {e}")

        # B: Match gegen bestehende PROTEIN-Nodes im Graph
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN"]
        detected = 0

        for node_id, protein in protein_nodes:
            accession = protein.get("id", "")
            desc = (protein.get("description") or "").lower()

            # DETECTION: UniProt-Annotation ODER Beschreibung enthält Allergen-Signale
            is_allergen = (
                accession in allergen_accessions
                or "allergen" in desc
                or "ige-binding" in desc
                or "ige binding" in desc
            )
            if not is_allergen:
                continue

            allergen_id = f"ALLERGEN_{accession}"
            self.g.add_node({
                "id": allergen_id,
                "type": "ALLERGEN",
                "label": protein.get("label", accession),
                "source_protein": accession,
                "description": protein.get("description"),
                "detection_method": "UNIPROT_KW0020" if accession in allergen_accessions else "DESCRIPTION_SCAN",
            })
            self.g.add_edge(
                src=accession, trgt=allergen_id,
                attrs={
                    "rel": "IS_ALLERGEN",
                    "src_layer": "PROTEIN",
                    "trgt_layer": "ALLERGEN",
                },
            )
            detected += 1

        print(f"Phase 14: {detected} allergen proteins detected "
              f"({len(allergen_accessions)} UniProt KW-0020 matches)")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 15: ALLERGEN MOLECULAR IMPACT (CTD + Open Targets)
    # A) CTD: Welche Gene werden durch das Allergen in der Expression
    #    verändert? -> Zytokin-Haushalt (IL-4, IL-5, IL-13 etc.)
    # B) Open Targets GraphQL: Immunologische Krankheitsassoziationen
    #    -> ALLERGEN -[TRIGGERS]-> IMMUNE_RESPONSE
    # ═══════════════════════════════════════════════════════════════════

    _OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    _OT_DISEASE_QUERY = """
    query($ensgId: String!) {
      target(ensemblId: $ensgId) {
        approvedSymbol
        associatedDiseases(page: {size: 50, index: 0}) {
          rows {
            disease { id name }
            score
          }
        }
      }
    }
    """

    async def enrich_allergen_molecular_impact(self):
        """
        Molecular Mapping: CTD Chemical-Gene Interactions + Open Targets Disease Links.
        Erzeugt ALTERS_EXPRESSION und TRIGGERS Kanten für den Allergie-Subgraph.
        """
        allergen_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                          if v.get("type") == "ALLERGEN"]

        if not allergen_nodes:
            print("Phase 15: No allergen nodes found, skipping")
            return

        ctd_links = 0
        immune_links = 0

        for allergen_id, allergen in allergen_nodes:
            allergen_label = allergen.get("label", "")
            source_protein = allergen.get("source_protein", "")

            # ── A: CTD Chemical-Gene Interactions ──────────────────────
            # ZIEL: welche Gene werden durch dieses Allergen hochreguliert?
            ctd_url = (
                f"http://ctdbase.org/tools/batchQuery.go"
                f"?inputType=chem&inputTerms={quote(allergen_label)}"
                f"&report=genes_curated&format=json"
            )
            try:
                ctd_res = await self.client.get(ctd_url, timeout=30.0)
                if ctd_res.status_code == 200:
                    content_type = ctd_res.headers.get("content-type", "")
                    if "json" in content_type:
                        raw = ctd_res.json()
                        interactions = raw if isinstance(raw, list) else []

                        for inter in interactions:
                            target_gene = inter.get("GeneSymbol")
                            action = inter.get("InteractionActions", "")
                            organism = inter.get("Organism", "")

                            # NUR humane Interaktionen verwerten
                            if "Homo sapiens" not in organism:
                                continue

                            gene_node_id = f"GENE_{target_gene}"
                            if not self.g.G.has_node(gene_node_id):
                                continue

                            # IMPACT: entzündlich wenn Expression steigt, regulatorisch sonst
                            impact = "INFLAMMATORY_CASCADE" if "increases" in action.lower() else "REGULATORY"

                            self.g.add_edge(
                                src=allergen_id,
                                trgt=gene_node_id,
                                attrs={
                                    "rel": "ALTERS_EXPRESSION",
                                    "mechanism": action,
                                    "impact": impact,
                                    "src_layer": "ALLERGEN",
                                    "trgt_layer": "GENE",
                                },
                            )
                            ctd_links += 1
            except Exception as e:
                print(f"CTD Error for {allergen_label}: {e}")

            # ── B: Open Targets Disease Associations (GraphQL) ────────
            # Benötigt Ensembl-ID via verlinktem GENE-Node
            ensembl_id = None
            if self.g.G.has_node(source_protein):
                for neighbor in self.g.G.neighbors(source_protein):
                    n_data = self.g.G.nodes.get(neighbor, {})
                    if n_data.get("type") == "GENE" and n_data.get("ensembl_id"):
                        ensembl_id = n_data["ensembl_id"]
                        break

            if not ensembl_id:
                continue

            try:
                ot_res = await self.client.post(
                    self._OT_URL,
                    json={"query": self._OT_DISEASE_QUERY, "variables": {"ensgId": ensembl_id}},
                    timeout=20.0,
                )
                if ot_res.status_code != 200:
                    continue

                target_data = ot_res.json().get("data", {}).get("target", {})
                rows = (target_data.get("associatedDiseases") or {}).get("rows", [])

                for row in rows:
                    disease = row.get("disease", {})
                    disease_name = disease.get("name", "")
                    disease_id = disease.get("id", "")
                    score = row.get("score", 0)

                    # FILTER: immunologisch relevante Assoziationen mit Score > 0.3
                    disease_lower = disease_name.lower()
                    is_immune = any(t in disease_lower for t in (
                        "allerg", "hypersensitiv", "asthma", "dermatitis",
                        "rhinitis", "anaphyla", "urticaria", "eczema",
                        "atopic", "immun", "inflammat", "histamin", "mast cell",
                    ))
                    if not is_immune or score < 0.3:
                        continue

                    response_id = f"IMMUNE_{disease_id.replace(':', '_')}"
                    if not self.g.G.has_node(response_id):
                        self.g.add_node({
                            "id": response_id,
                            "type": "IMMUNE_RESPONSE",
                            "label": disease_name,
                            "disease_id": disease_id,
                            "association_score": score,
                        })

                    self.g.add_edge(
                        src=allergen_id, trgt=response_id,
                        attrs={
                            "rel": "TRIGGERS",
                            "score": score,
                            "src_layer": "ALLERGEN",
                            "trgt_layer": "IMMUNE",
                        },
                    )
                    immune_links += 1

            except Exception as e:
                print(f"Open Targets Error for {allergen_label}: {e}")

        print(f"Phase 15: {ctd_links} gene expression links (CTD), "
              f"{immune_links} immune response links (Open Targets)")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 16: ALLERGEN-FOOD CROSS-LINKING + KREUZALLERGIE
    # Verbindet FOOD_SOURCE -> ALLERGEN wenn ein Lebensmittel ein Protein
    # enthält, das als Allergen markiert ist. Domänen-Überlappung zwischen
    # Allergenen erzeugt CROSS_REACTIVITY Kanten (Kreuzallergie-Prädiktion).
    # ═══════════════════════════════════════════════════════════════════

    def crosslink_allergen_food_sources(self):
        """
        Cross-Link: FOOD_SOURCE -> ALLERGEN + CROSS_REACTIVITY zwischen Allergenen.
        Ermöglicht Vorhersage, welche Lebensmittel ähnliche Entzündungskaskaden auslösen.
        """
        # LOOKUP: source_protein_accession -> allergen_node_id
        allergen_by_protein: dict[str, str] = {
            v.get("source_protein"): k
            for k, v in self.g.G.nodes(data=True)
            if v.get("type") == "ALLERGEN" and v.get("source_protein")
        }

        if not allergen_by_protein:
            print("Phase 16: No allergen nodes for cross-linking")
            return

        # ── A: FOOD_SOURCE -> ALLERGEN via CONTAINS_NUTRIENT Kanten ──
        linked = 0
        for u, v, data in self.g.G.edges(data=True):
            if data.get("rel") != "CONTAINS_NUTRIENT":
                continue
            if self.g.G.nodes.get(u, {}).get("type") != "FOOD_SOURCE":
                continue

            allergen_id = allergen_by_protein.get(v)
            if not allergen_id:
                continue

            self.g.add_edge(
                src=u, trgt=allergen_id,
                attrs={
                    "rel": "CONTAINS_ALLERGEN",
                    "severity": "CRITICAL",
                    "src_layer": "FOOD",
                    "trgt_layer": "ALLERGEN",
                },
            )
            linked += 1
            food_label = self.g.G.nodes.get(u, {}).get("label", u)
            allergen_label = self.g.G.nodes.get(allergen_id, {}).get("label", allergen_id)
            print(f"Critical Path: {food_label} -> ALLERGEN {allergen_label}")

        # ── B: KREUZALLERGIE via gemeinsame Protein-Domänen ──────────
        # Wenn zwei Allergene dieselbe Domäne teilen -> Kreuzreaktivitätsrisiko
        allergen_domains: dict[str, set[str]] = {}
        for protein_acc, a_id in allergen_by_protein.items():
            domains: set[str] = set()
            if self.g.G.has_node(protein_acc):
                for _, neighbor, edata in self.g.G.edges(protein_acc, data=True):
                    if edata.get("rel") == "CONTAINS_DOMAIN":
                        domains.add(neighbor)
            if domains:
                allergen_domains[a_id] = domains

        cross_links = 0
        allergen_list = list(allergen_domains.keys())
        for i, a_id in enumerate(allergen_list):
            for b_id in allergen_list[i + 1:]:
                shared = allergen_domains[a_id] & allergen_domains[b_id]
                if shared:
                    self.g.add_edge(
                        src=a_id, trgt=b_id,
                        attrs={
                            "rel": "CROSS_REACTIVITY",
                            "shared_domains": list(shared),
                            "domain_overlap": len(shared),
                            "src_layer": "ALLERGEN",
                            "trgt_layer": "ALLERGEN",
                        },
                    )
                    cross_links += 1

        print(f"Phase 16: {linked} food-allergen links, {cross_links} cross-reactivity pairs")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 17: CELLULAR COMPONENTS + CODING / NON-CODING GENE MAPPING
    # A) UniProt cc_subcellular_location -> CELLULAR_COMPONENT nodes
    #    linked to PROTEIN and coding GENE nodes.
    # B) Ensembl overlap -> NON_CODING_GENE nodes (lncRNA, miRNA, …)
    #    linked to same CELLULAR_COMPONENT + nearby coding GENE.
    # ═══════════════════════════════════════════════════════════════════

    async def enrich_cellular_components(self):
        """
        Two-pass cellular-component integration:
          A) PROTEIN -> CELLULAR_COMPONENT via UniProt cc_subcellular_location
             + GENE (coding) -> CELLULAR_COMPONENT back-link
          B) Ensembl overlap -> NON_CODING_GENE -> CELLULAR_COMPONENT + GENE
        """
        protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                         if v.get("type") == "PROTEIN"]
        comp_count = 0
        coding_links = 0

        # ── PASS A: UniProt subcellular locations ─────────────────────
        for node_id, protein in protein_nodes:
            accession = protein.get("id")
            if not accession:
                continue

            url = (
                f"https://rest.uniprot.org/uniprotkb/{accession}"
                f"?fields=cc_subcellular_location&format=json"
            )
            try:
                res = await self.fetch_with_retry(url)
                comments = res.get("comments", [])

                # COLLECT: components this protein resides in (for Pass B back-ref)
                protein_comp_ids: list[str] = []

                for comment in comments:
                    if comment.get("commentType") != "SUBCELLULAR LOCATION":
                        continue

                    for sl in comment.get("subcellularLocations", []):
                        loc = sl.get("location", {})
                        sl_id = loc.get("id")
                        sl_label = loc.get("value")
                        if not sl_id or not sl_label:
                            continue

                        comp_node_id = f"CELLCOMP_{sl_id}"

                        # IDEMPOTENT: create component node once
                        if not self.g.G.has_node(comp_node_id):
                            topo = sl.get("topology", {})
                            self.g.add_node({
                                "id": comp_node_id,
                                "type": "CELLULAR_COMPONENT",
                                "label": sl_label,
                                "sl_id": sl_id,
                                "topology": topo.get("value"),
                            })
                            comp_count += 1

                        # ECO: best evidence score from this location's evidences
                        best_rel = 0.0
                        best_eco = None
                        for ev in loc.get("evidences", []):
                            eco_code = ev.get("evidenceCode")
                            rel, _ = ECO_RELIABILITY.get(eco_code, ECO_DEFAULT)
                            if rel > best_rel:
                                best_rel, best_eco = rel, eco_code

                        self.g.add_edge(
                            src=node_id, trgt=comp_node_id,
                            attrs={
                                "rel": "RESIDES_IN",
                                "eco_code": best_eco,
                                "reliability": best_rel,
                                "src_layer": "PROTEIN",
                                "trgt_layer": "CELLULAR_COMPONENT",
                            },
                        )
                        protein_comp_ids.append(comp_node_id)

                # BACK-LINK: coding GENE -> CELLULAR_COMPONENT
                for neighbor in self.g.G.neighbors(node_id):
                    n_data = self.g.G.nodes.get(neighbor, {})
                    if n_data.get("type") != "GENE":
                        continue
                    for cid in protein_comp_ids:
                        self.g.add_edge(
                            src=neighbor, trgt=cid,
                            attrs={
                                "rel": "CODING_GENE_IN_COMPONENT",
                                "src_layer": "GENE",
                                "trgt_layer": "CELLULAR_COMPONENT",
                            },
                        )
                        coding_links += 1

            except Exception as e:
                print(f"CellComp Error for {accession}: {e}")

        print(f"Pass A done: {comp_count} CELLULAR_COMPONENT nodes, "
              f"{coding_links} coding-gene links")

        # ── PASS B: Non-coding genes via Ensembl overlap ──────────────
        gene_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                      if v.get("type") == "GENE"
                      and v.get("chromosome")
                      and v.get("gene_start") is not None
                      and v.get("gene_end") is not None]

        _seen_nc: set[str] = set()
        nc_count = 0
        nc_comp_links = 0

        for gene_id, gene in gene_nodes:
            if nc_count >= _MAX_NC_GENE_NODES:
                print(f"NON_CODING_GENE cap reached ({_MAX_NC_GENE_NODES})")
                break

            chrom = gene["chromosome"]
            start = max(1, gene["gene_start"] - _OVERLAP_FLANK_BP)
            end = gene["gene_end"] + _OVERLAP_FLANK_BP
            region = f"{chrom}:{start}-{end}"

            # BIOTYPE FILTER: only non-coding classes
            biotype_params = ";".join(f"biotype={bt}" for bt in _NC_BIOTYPES)
            ensembl_url = (
                f"https://rest.ensembl.org/overlap/region/homo_sapiens/{region}"
                f"?feature=gene;{biotype_params};content-type=application/json"
            )

            try:
                overlap_res = await self.client.get(ensembl_url, timeout=20.0)
                if overlap_res.status_code != 200:
                    continue
                nc_genes = overlap_res.json()
                if not isinstance(nc_genes, list):
                    continue

                for nc in nc_genes:
                    nc_ensembl = nc.get("id") or nc.get("gene_id")
                    if not nc_ensembl or nc_ensembl in _seen_nc:
                        continue
                    if nc_count >= _MAX_NC_GENE_NODES:
                        break

                    nc_node_id = f"NCGENE_{nc_ensembl}"
                    nc_label = (nc.get("external_name")
                                or nc.get("Name")
                                or nc_ensembl)
                    nc_biotype = nc.get("biotype", "unknown")

                    self.g.add_node({
                        "id": nc_node_id,
                        "type": "NON_CODING_GENE",
                        "label": nc_label,
                        "biotype": nc_biotype,
                        "ensembl_id": nc_ensembl,
                        "chromosome": chrom,
                        "start": nc.get("start"),
                        "end": nc.get("end"),
                    })
                    _seen_nc.add(nc_ensembl)
                    nc_count += 1

                    # EDGE: NON_CODING_GENE -> nearby coding GENE
                    self.g.add_edge(
                        src=nc_node_id, trgt=gene_id,
                        attrs={
                            "rel": "OVERLAPS_CODING_GENE",
                            "src_layer": "NON_CODING",
                            "trgt_layer": "GENE",
                        },
                    )

                    # INFERRED LOCALIZATION: share coding gene's components
                    for _, neighbor, edata in self.g.G.edges(gene_id, data=True):
                        if edata.get("rel") == "CODING_GENE_IN_COMPONENT":
                            self.g.add_edge(
                                src=nc_node_id, trgt=neighbor,
                                attrs={
                                    "rel": "NC_GENE_IN_COMPONENT",
                                    "inferred_from": gene_id,
                                    "src_layer": "NON_CODING",
                                    "trgt_layer": "CELLULAR_COMPONENT",
                                },
                            )
                            nc_comp_links += 1

            except Exception as e:
                print(f"Ensembl Overlap Error for {gene.get('label')}: {e}")

        print(f"Pass B done: {nc_count} NON_CODING_GENE nodes, "
              f"{nc_comp_links} component links (inferred)")

    # --- GRAPH EMBEDDING & TEMPSTORE ─────────────────────────────────

    # EMBED_MODEL: Gemini text-embedding-004 (768-dim, best retrieval quality)
    _EMBED_MODEL = "models/text-embedding-004"
    _EMBED_BATCH = 96  # API max per request ~100, keep headroom

    @staticmethod
    def _node_to_text(node_id: str, attrs: dict) -> str:
        """Deterministic text repr of a node for embedding."""
        parts = [f"[{attrs.get('type', 'NODE')}]", f"id={node_id}"]
        for key in ("label", "description", "smiles", "evidence_type",
                     "ion_selectivity", "metabolic_role", "cl_id",
                     "pLDDT_avg", "interpro_id", "domain_type",
                     "go_id", "aspect", "definition", "sequence_hash", "full_hash",
                     "detection_method", "source_protein",
                     "disease_id", "association_score",
                     "sl_id", "topology", "biotype",
                     "activity_relation"):
            val = attrs.get(key)
            if val:
                parts.append(f"{key}={val}")
        # LIST FIELDS: synonyms (GO_TERM), evidence_types (GOCAM_ACTIVITY)
        for list_key in ("synonyms", "evidence_types"):
            vals = attrs.get(list_key)
            if vals and isinstance(vals, list):
                parts.append(f"{list_key}={';'.join(str(v) for v in vals[:5])}")
        return " | ".join(parts)

    @staticmethod
    def _edge_to_text(src: str, trgt: str, attrs: dict) -> str:
        """Deterministic text repr of an edge for embedding."""
        rel = attrs.get("rel", "RELATED_TO")
        parts = [f"[EDGE:{rel}]", f"{src} -> {trgt}"]
        for key in ("src_layer", "trgt_layer", "action", "mechanism",
                     "causality", "source_type",
                     "impact", "severity", "score", "domain_overlap",
                     "qualifier", "assigned_by", "derived_from",
                     "reliability", "evidence_type"):
            val = attrs.get(key)
            if val:
                parts.append(f"{key}={val}")
        return " | ".join(parts)

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Gemini batch embedding with chunked requests to stay under API limits."""
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._EMBED_BATCH):
            chunk = texts[i : i + self._EMBED_BATCH]
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self._EMBED_MODEL,
                content=chunk,
                task_type="RETRIEVAL_DOCUMENT",
            )
            all_vectors.extend(result["embedding"])
        return all_vectors

    async def embed_graph_to_tempstore(self) -> str:
        """
        EMBED every node & edge -> write plain graph + embedded graph to tempdir.
        Returns the tempdir path containing:
          graph.json            – raw NetworkX node-link export
          graph.embedded.json   – same structure with '_embedding' on each element
        """
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        G: nx.MultiGraph = self.g.G

        # ── 1. Serialise raw graph ──────────────────────────────────────
        raw_data = nx.node_link_data(G)

        # ── 2. Collect texts for nodes ──────────────────────────────────
        node_ids: list[str] = []
        node_texts: list[str] = []
        for nid, attrs in G.nodes(data=True):
            node_ids.append(nid)
            node_texts.append(self._node_to_text(nid, attrs))

        # ── 3. Collect texts for edges ──────────────────────────────────
        edge_keys: list[tuple] = []
        edge_texts: list[str] = []
        for src, trgt, key, attrs in G.edges(data=True, keys=True):
            edge_keys.append((src, trgt, key))
            edge_texts.append(self._edge_to_text(src, trgt, attrs))

        # ── 4. Batch-embed everything in one pass ───────────────────────
        all_texts = node_texts + edge_texts
        if not all_texts:
            print("EMBED: graph empty, nothing to embed")
            return ""

        print(f"EMBED: encoding {len(node_texts)} nodes + {len(edge_texts)} edges …")
        all_vectors = await self._batch_embed(all_texts)

        node_vectors = all_vectors[: len(node_texts)]
        edge_vectors = all_vectors[len(node_texts) :]

        # ── 5. Build embedded copy ──────────────────────────────────────
        embedded_data = nx.node_link_data(G)

        # ATTACH node embeddings
        nid_to_vec = dict(zip(node_ids, node_vectors))
        for node_entry in embedded_data["nodes"]:
            vec = nid_to_vec.get(node_entry["id"])
            if vec:
                node_entry["_embedding"] = vec

        # ATTACH edge embeddings
        ekey_to_vec = dict(zip(edge_keys, edge_vectors))
        for link_entry in embedded_data["links"]:
            k = (link_entry.get("source"), link_entry.get("target"), link_entry.get("key"))
            vec = ekey_to_vec.get(k)
            if vec:
                link_entry["_embedding"] = vec

        # ── 6. Write to tempstore ───────────────────────────────────────
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        store_dir = tempfile.mkdtemp(prefix=f"acid_graph_{ts}_")

        raw_path = os.path.join(store_dir, "graph.json")
        emb_path = os.path.join(store_dir, "graph.embedded.json")

        # INTEGRITY: sha256 of raw for downstream validation
        raw_bytes = json.dumps(raw_data, ensure_ascii=False, default=str).encode()
        with open(raw_path, "wb") as f:
            f.write(raw_bytes)

        emb_bytes = json.dumps(embedded_data, ensure_ascii=False, default=str).encode()
        with open(emb_path, "wb") as f:
            f.write(emb_bytes)

        # MANIFEST with checksums
        manifest = {
            "created_utc": ts,
            "node_count": len(node_ids),
            "edge_count": len(edge_keys),
            "embedding_model": self._EMBED_MODEL,
            "embedding_dim": len(node_vectors[0]) if node_vectors else 0,
            "sha256_raw": hashlib.sha256(raw_bytes).hexdigest(),
            "sha256_embedded": hashlib.sha256(emb_bytes).hexdigest(),
        }
        with open(os.path.join(store_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"EMBED: stored -> {store_dir}")
        print(f"  graph.json          ({len(raw_bytes):,} bytes)")
        print(f"  graph.embedded.json ({len(emb_bytes):,} bytes)")
        return store_dir

    # --- PHASE 18: ELECTRON DENSITY MATRIX (RDKit + PySCF) ─────────
    # SMILES → 3D-Atome → Einteilchen-Dichtematrix (RDM1)
    _PARENT_LAYER_TYPES = frozenset({
        "PHARMA_COMPOUND", "PROTEIN", "MINERAL", "MOLECULE_CHAIN", "GENE",
    })
    # PHYSIK-KONSTANTEN für Anregungsenergie → Laser-Parameter
    _HA_TO_EV = 27.211386
    _EV_TO_NM = 1239.84193
    _EV_TO_HZ = 2.417989242e14
    _NIR_LOW = 650    # nm – untere Grenze therapeutisches Fenster
    _NIR_HIGH = 900   # nm – obere Grenze therapeutisches Fenster
    _MIN_OSC_STRENGTH = 0.001  # Schwelle für messbare Absorption

    def compute_electron_density_matrices(self):
        """
        Übersetzt alle Molekül-Nodes (SMILES) in Atome, berechnet die
        Einteilchen-Dichtematrix via DFT(B3LYP)/def2-SVP + ddCOSMO(Wasser),
        und bestimmt per TD-DFT die Anregungsenergien (Laser-Frequenzen).
        Pro messbarer Anregung wird ein EXCITATION_FREQUENCY Node erstellt.
        Ergebnisse + Frequenzen werden an alle Eltern-Layer geheftet.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from pyscf import gto, dft, tddft, solvent
            import numpy as np
        except ImportError as exc:
            print(f"PHASE 18 SKIP – fehlende Abhängigkeit: {exc}")
            return

        # ── ALLE NODES MIT GÜLTIGEM SMILES SAMMELN ──────────────────
        smiles_nodes = [
            (nid, data) for nid, data in self.g.G.nodes(data=True)
            if data.get("smiles") and data["smiles"] not in (None, "N/A")
        ]
        if not smiles_nodes:
            print("  Keine Nodes mit SMILES gefunden – überspringe.")
            return

        computed, skipped = 0, 0

        for node_id, node in smiles_nodes:
            smiles = node["smiles"]

            # ── SCHRITT 1: RDKit – SMILES → 3D-Koordinaten ──────────
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  RDKit parse-Fehler: {smiles[:60]} – skip")
                skipped += 1
                continue

            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
                print(f"  3D-Embedding fehlgeschlagen: {smiles[:60]} – skip")
                skipped += 1
                continue
            AllChem.MMFFOptimizeMolecule(mol)

            conf = mol.GetConformer()
            atoms_pyscf = []
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                atoms_pyscf.append([atom.GetSymbol(), [pos.x, pos.y, pos.z]])

            # ── SCHRITT 2: DFT(B3LYP) + ddCOSMO(Wasser) für in-vivo Relevanz
            try:
                pyscf_mol = gto.Mole(verbose=0)
                pyscf_mol.atom = atoms_pyscf
                pyscf_mol.basis = "def2-SVP"
                pyscf_mol.build()

                mf = dft.RKS(pyscf_mol)
                mf.xc = "B3LYP"
                mf = solvent.ddCOSMO(mf)
                mf.with_solvent.eps = 78.3553  # Wasser-Dielektrikum (in vivo)
                mf.kernel()

                rdm1 = mf.make_rdm1()
            except Exception as exc:
                print(f"  PySCF-Fehler für {node_id}: {exc} – skip")
                skipped += 1
                continue

            # ── SCHRITT 3: Ergebnisse auf Node + Eltern-Layer ────────
            n_basis = rdm1.shape[0]
            # OBERES DREIECK (symmetrisch) als flache Liste → JSON-tauglich
            upper_tri = rdm1[np.triu_indices(n_basis)].tolist()

            node["atom_decomposition"] = atoms_pyscf
            node["electron_density_matrix"] = upper_tri
            node["density_matrix_shape"] = [n_basis, n_basis]
            node["density_matrix_basis"] = "def2-SVP"
            node["total_electrons"] = float(np.trace(rdm1))
            node["total_energy_hartree"] = float(mf.e_tot)
            node["scf_converged"] = bool(mf.converged)

            # PROPAGATION: Summary an übergeordnete Layer-Nodes heften
            density_summary = {
                "source_node": node_id,
                "total_electrons": node["total_electrons"],
                "total_energy_hartree": node["total_energy_hartree"],
                "basis": "def2-SVP",
                "scf_converged": node["scf_converged"],
            }
            for neighbor in self.g.G.neighbors(node_id):
                nb_data = self.g.G.nodes[neighbor]
                if nb_data.get("type") in self._PARENT_LAYER_TYPES:
                    nb_data.setdefault("electron_density_children", []).append(density_summary)

            # ── SCHRITT 4: TD-DFT → Anregungsenergien + Laser-Frequenzen ─
            src_layer = node.get("type", "ATOMIC_STRUCTURE")
            try:
                td = mf.TDDFT()
                td.nstates = 10
                td.kernel()

                exc_energies_ev = td.e * self._HA_TO_EV
                osc_strengths = td.oscillator_strength(gauge='length')

                for state_idx, (e_ev, f_osc) in enumerate(
                    zip(exc_energies_ev, osc_strengths), start=1
                ):
                    if f_osc < self._MIN_OSC_STRENGTH:
                        continue

                    wl_nm = self._EV_TO_NM / e_ev
                    freq_hz = e_ev * self._EV_TO_HZ

                    # EXCITATION_FREQUENCY Node
                    freq_id = f"FREQ_{node_id}_S{state_idx}"
                    self.g.add_node({
                        "id": freq_id,
                        "type": "EXCITATION_FREQUENCY",
                        "label": f"S0->S{state_idx} {wl_nm:.1f}nm",
                        "excitation_energy_ev": round(float(e_ev), 6),
                        "wavelength_nm": round(float(wl_nm), 2),
                        "frequency_hz": float(freq_hz),
                        "oscillator_strength": round(float(f_osc), 6),
                        "in_nir_window": bool(self._NIR_LOW <= wl_nm <= self._NIR_HIGH),
                        "solvent": "water",
                        "basis": "def2-SVP",
                        "xc_functional": "B3LYP",
                    })

                    # Edge: Quell-SMILES-Node → EXCITATION_FREQUENCY
                    self.g.add_edge(
                        src=node_id, trgt=freq_id,
                        attrs={
                            "rel": "HAS_EXCITATION",
                            "src_layer": src_layer,
                            "trgt_layer": "PHOTOPHYSICS",
                        },
                    )

                    # Frequenz-Summary an alle Eltern-Layer-Nodes heften
                    freq_summary = {
                        "source_node": node_id,
                        "freq_node": freq_id,
                        "wavelength_nm": round(float(wl_nm), 2),
                        "oscillator_strength": round(float(f_osc), 6),
                        "in_nir_window": bool(self._NIR_LOW <= wl_nm <= self._NIR_HIGH),
                    }
                    for neighbor in self.g.G.neighbors(node_id):
                        nb_data = self.g.G.nodes[neighbor]
                        if nb_data.get("type") in self._PARENT_LAYER_TYPES:
                            nb_data.setdefault("excitation_frequency_children", []).append(freq_summary)

                    nir_tag = " [NIR]" if self._NIR_LOW <= wl_nm <= self._NIR_HIGH else ""
                    print(f"    FREQ S{state_idx}: {wl_nm:.1f} nm  f={f_osc:.4f}{nir_tag}")

            except Exception as exc:
                print(f"  TD-DFT-Fehler für {node_id}: {exc} – Frequenzen übersprungen")

            computed += 1
            print(f"  RDM1+TD OK: {node_id}  ({n_basis}x{n_basis}, E={mf.e_tot:.6f} Ha)")

        print(f"  Fertig – {computed} berechnet, {skipped} übersprungen.")

    # --- CONSOLIDATED WORKFLOW ---
    async def finalize_biological_graph(self):
        """Zentraler Orchestrator für den kompletten Graphaufbau."""
        try:
            print("--- PHASE 1: Initial Protein & Gene Ingestion ---")
            await self.get_all_proteins()

            print("--- PHASE 2: Deep Fetching UniProt Details ---")
            await self.enrich_gene_nodes_deep()

            print("--- PHASE 3: Live Pharmacology (ChEMBL + BfArM) ---")
            await self.enrich_pharmacology_quantum_adme()

            print("--- PHASE 4: Atomic & Molecular Mapping (SMILES) ---")
            await self.enrich_molecular_structures()

            print("--- PHASE 5: Nutritional Origin (Open Food Facts DE) ---")
            await self.enrich_food_sources()

            print("--- PHASE 6: Genomic & Functional Enrichment ---")
            await asyncio.gather(
                self.enrich_genomic_data(),
                self.enrich_functional_dynamics(),
            )

            print("--- PHASE 7: Pharmacogenomics (ClinPGx) ---")
            await self.enrich_pharmacogenomics()

            print("--- PHASE 8: Bioelectric Properties (GtoPdb) ---")
            await self.enrich_bioelectric_properties()

            print("--- PHASE 9: Microbiome Metabolism (VMH) ---")
            await self.enrich_microbiome_axis()

            print("--- PHASE 10: Cellular Integration (HPA + Cell Ontology) ---")
            await self.enrich_cell_type_expression()

            # ── STRUCTURAL-FUNCTIONAL-KNOWLEDGE-GRAPH LAYER ──────────
            print("--- PHASE 10.5: Sequence Identity Hashing (SHA-256) ---")
            self.compute_sequence_hashes()

            print("--- PHASE 11: Structural Inference (AlphaFold DB) ---")
            await self.enrich_structural_layer()

            print("--- PHASE 12: Domain Decomposition (InterPro) ---")
            await self.enrich_domain_decomposition()

            print("--- PHASE 13a: GO-Semantic-Linking (QuickGO) ---")
            await self.enrich_go_semantic_layer()

            print("--- PHASE 13a+: GO Term Metadata Enrichment ---")
            await self._enrich_go_term_metadata()

            print("--- PHASE 13a++: GO Ontology Hierarchy ---")
            await self._wire_go_hierarchy()

            print("--- PHASE 13a+++: GENE -> GO_TERM Derived Edges ---")
            self._wire_gene_go_edges()

            print("--- PHASE 13b: Subcellular Localization (COMPARTMENTS) ---")
            await self.enrich_compartment_localization()

            print("--- PHASE 13c: GO-CAM Causal Activity Models ---")
            await self.enrich_gocam_activities()

            # ── ALLERGIE-MOLEKULAR-GRAPH ───────────────────────────────
            print("--- PHASE 14: Allergen Detection (UniProt KW-0020) ---")
            await self.detect_allergen_proteins()

            print("--- PHASE 15: Allergen Molecular Impact (CTD + Open Targets) ---")
            await self.enrich_allergen_molecular_impact()

            print("--- PHASE 16: Allergen-Food Cross-Linking & Kreuzallergie ---")
            self.crosslink_allergen_food_sources()

            print("--- PHASE 17: Cellular Components + Coding/Non-Coding Gene Mapping ---")
            await self.enrich_cellular_components()

            # ── QUANTEN-CHEMISCHE DICHTEMATRIX ────────────────────────
            print("--- PHASE 18: Electron Density Matrix (RDKit + PySCF) ---")
            self.compute_electron_density_matrices()

            self.g.print_status_G()
        finally:
            await self.close()





if __name__ == "__main__":
    g = GUtils()
    kb = UniprotKB(g)

    # Starte den konsolidierten asynchronen Workflow
    asyncio.run(kb.finalize_biological_graph())