import numpy as np
from tqdm import tqdm
from pronto import Ontology
from embedder import embed, embed_batch


def build_brain_tissue_subgraph(
    g,
    tissue: str,
    uberon_path: str = "tissue/uberon.obo",
    threshold: float = 0.90,
):
    print(f"Loading ontology: {uberon_path}")

    ont = Ontology(uberon_path)
    query_vec = embed(tissue)

    terms = []
    texts = []

    print("Collecting Uberon terms...")

    for term in ont.terms():

        if not str(term.id).startswith("UBERON:"):
            continue

        text = term.name or ""

        if getattr(term, "definition", None):
            text += f" {term.definition}"

        terms.append(term)
        texts.append(text)

    print(f"Collected {len(terms):,} Uberon terms")

    print("Embedding ontology terms...")

    term_vecs = embed_batch(texts)

    query_vec = np.asarray(query_vec).reshape(1, -1)
    term_vecs = np.asarray(term_vecs)

    query_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
    term_norms = np.linalg.norm(term_vecs, axis=1, keepdims=True)

    similarities = (
        np.dot(term_vecs, query_vec.T)
        / (term_norms * query_norm)
    ).flatten()

    matched_ids = set()

    print("Adding matched nodes...")

    for term, score, text in zip(terms, similarities, texts):

        if score < threshold:
            continue

        matched_ids.add(term.id)

        g.add_node(
            attrs=dict(
                id=term.id,
                type="UBERON",
                text=text,
                #score=float(score),
                embed_key="text",
            )
        )
    print(f"Matched {len(matched_ids):,} Uberon terms")

    print("Creating hierarchy edges...")
    for term_id in tqdm(matched_ids):

        term = ont[term_id]

        for child in term.subclasses(distance=1):
            print("working UBERON child", child)

            g.add_node(
                attrs=dict(
                    id=child.id,
                    type="UBERON",
                    text=child.name,
                    embed_key="text",
                )
            )

            g.add_edge(
                child.id,
                term.id,
                attr=dict(
                    rel="IS_A",
                    src_layer="UBERON",
                    trgt_layer="UBERON",
                ),
            )
    print("Done")
    return list(matched_ids)