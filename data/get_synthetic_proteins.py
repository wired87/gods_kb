import requests

from data.protein_disease import add_disease_node
from embedder import embed


def get_synthetic_proteins_for_human(g, target_context="human brain"):
    """
    Holt synthetische Konstrukte (TaxID 32630), die im Kontext von
    'human' oder 'human brain' stehen.
    """
    print(f"\n[START] ingest_synthetic_proteins: {target_context}")

    url = "https://rest.uniprot.org/uniprotkb/search"
    query = f'taxonomy_id:32630 AND "{target_context}"'

    params = {
        "query": query,
        "format": "json",
        "size": 500
    }

    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()

    for entry in data["results"]:
        acc = entry["primaryAccession"]
        node_id = f"{acc}"

        # -------------------------
        # 🧠 CORE PROTEIN NODE
        # -------------------------
        gene = entry.get("genes", [{}])[0].get("geneName", {}).get("value")
        protein_name = (
            entry.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value")
        )
        # -------------------------
        # 🧠 CORE PROTEIN NODE (FULL AGGREGATION)
        # -------------------------

        gene_names = []

        for g in entry.get("genes", []):
            name = g.get("geneName", {}).get("value")
            if name:
                gene_names.append(name)

            for syn in g.get("synonyms", []):
                if syn.get("value"):
                    gene_names.append(syn["value"])

        # ---- evidences sammeln ----
        evidence_ids = []

        def _collect_evidences(obj):
            if isinstance(obj, dict):
                if "evidences" in obj:
                    for e in obj["evidences"]:
                        if "evidenceCode" in e:
                            evidence_ids.append(str(e["evidenceCode"]))
                for v in obj.values():
                    _collect_evidences(v)

            elif isinstance(obj, list):
                for i in obj:
                    _collect_evidences(i)

        _collect_evidences(entry)

        # ---- cross refs sammeln ----
        cross_refs = []
        cross_refs_by_db = {}

        for ref in entry.get("uniProtKBCrossReferences", []):
            db = ref.get("database")
            rid = ref.get("id")

            if rid:
                cross_refs.append(rid)

                if db:
                    cross_refs_by_db.setdefault(db, []).append(rid)

        # ---- isoform ids ----
        isoform_ids = []
        for c in entry.get("comments", []):
            if c.get("commentType") == "ALTERNATIVE PRODUCTS":
                for iso in c.get("isoforms", []):
                    isoform_ids.extend(iso.get("isoformIds", []))

        # ---- locations ----
        locations = []
        for c in entry.get("comments", []):
            if c.get("commentType") == "SUBCELLULAR LOCATION":
                for loc in c.get("subcellularLocations", []):
                    l = loc.get("location", {}).get("value")
                    if l:
                        locations.append(l)

        # ---- diseases ----
        diseases = []
        for c in entry.get("comments", []):
            if c.get("commentType") == "DISEASE":
                d = c.get("disease", {})
                if d.get("diseaseAccession"):
                    diseases.append(d["diseaseAccession"])

        # ---- keywords ----
        keywords = [kw.get("name") for kw in entry.get("keywords", []) if kw.get("name")]

        # ---- features summary ----
        features = []
        for f in entry.get("features", []):
            features.append({
                "type": f.get("type"),
                "start": f.get("location", {}).get("start", {}).get("value"),
                "end": f.get("location", {}).get("end", {}).get("value"),
            })

        # ---- functions (text only) ----
        functions = []
        for c in entry.get("comments", []):
            if c.get("commentType") == "FUNCTION":
                for t in c.get("texts", []):
                    if t.get("value"):
                        functions.append(t["value"][:500])

        # -------------------------
        # FINAL PROTEIN NODE
        # -------------------------
        protein_node = {
            "id": node_id,
            "type": "PROTEIN",

            # core
            "accession": acc,
            "name": protein_name,

            "organism": entry.get("organism", {}).get("scientificName"),

            # quality
            "annotation_score": entry.get("annotationScore"),
            "protein_existence": entry.get("proteinExistence"),

            # 🔥 aggregated knowledge
            "functions": functions,
            "locations": list(set(locations)),
            "diseases": list(set(diseases)),
            "isoforms": list(set(isoform_ids)),
            "keywords": keywords,

            "functional_annotation": [embed(item) for item in functions],
            "outsrc_emb": [embed(item) for item in list(set(diseases))],

            # 🔬 structure
            "features": features,
        }
        g.add_node(protein_node)
        edges = {
            "genes": list(set(gene_names)),
            "evidence_ids": list(set(evidence_ids)),
            "cross_refs": list(set(cross_refs)),
            "cross_refs_by_db": cross_refs_by_db,
        }

        for etype, eid in edges.items():
            g.add_edge(
                src=node_id,
                trgt=eid,
                attrs=dict(
                    rel="ref",
                    src_layer="PROTEIN",
                    trgt_layer=etype.upper(),
                )
            )

        # DISEASE
        for disease_id in edges["disease"]:
            add_disease_node(
                g,
                accession=disease_id,
                protein_id=node_id,
            )
    print(f"[DONE] total proteins:")

if __name__ == "__main__":
    proteins = get_synthetic_proteins_for_human()
    print(proteins[0])