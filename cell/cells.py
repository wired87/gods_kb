"""
cell ref atlas zellen für tissue
cellxgene gene expression für tissue
"""










"""









def build_hpa_celltype_graph(
    g,
    file_path: str,
):

    print("Loading HPA file...")

    df = pd.read_csv(
        file_path,
        sep="\t",
        low_memory=False,
    )

    print(f"Loaded {len(df):,} rows")

    gene_cache = set()
    protein_cache = set()
    celltype_cache = set()

    for idx, row in enumerate(df.itertuples(index=False), start=1):

        if idx % 10000 == 0:
            print(f"Processed {idx:,} rows")

        #
        # column names may differ slightly between HPA releases
        #
        ens_id = getattr(row, "Gene", None)
        gene_name = getattr(row, "Gene_name", None)

        if gene_name is None:
            gene_name = getattr(row, "gene_name", None)

        cell_type = getattr(row, "Cell_type_group", None)

        if cell_type is None:
            cell_type = getattr(row, "cell_type_group", None)

        ncpm = getattr(row, "nCPM", None)

        if ncpm is None:
            ncpm = getattr(row, "nTPM", None)

        uniprot = getattr(row, "Uniprot", None)

        #
        # fallback for alternative HPA naming
        #
        if ens_id is None:
            ens_id = row[0]

        if gene_name is None:
            continue

        #
        # GENE
        #
        if ens_id not in gene_cache:

            g.add_node(
                dict(
                    id=ens_id,
                    name=gene_name,
                    type="GENE",
                    text=gene_name,
                    embed_key="text",
                )
            )

            gene_cache.add(ens_id)

        #
        # PROTEIN
        #
        if pd.notna(uniprot):

            protein_id = f"PROTEIN:{uniprot}"

            if protein_id not in protein_cache:

                g.add_node(
                    dict(
                        id=protein_id,
                        name=gene_name,
                        uniprot=uniprot,
                        type="PROTEIN",
                        text=gene_name,
                        embed_key="text",
                    )
                )

                protein_cache.add(protein_id)

            g.add_edge(
                src=ens_id,
                trgt=protein_id,
                attrs=dict(
                    rel="encodes",
                    src_layer="GENE",
                    trgt_layer="PROTEIN",
                )
            )

        #
        # CELL TYPE GROUP
        #
        if pd.notna(cell_type):

            cell_id = f"CELL_TYPE:{cell_type}"

            if cell_id not in celltype_cache:

                g.add_node(
                    dict(
                        id=cell_id,
                        name=cell_type,
                        type="CELL_TYPE_GROUP",
                        text=cell_type,
                        embed_key="text",
                    )
                )

                celltype_cache.add(cell_id)

            g.add_edge(
                src=cell_id,
                trgt=ens_id,
                attrs=dict(
                    rel="expresses",
                    nCPM=float(ncpm) if pd.notna(ncpm) else None,
                    src_layer="CELL_TYPE_GROUP",
                    trgt_layer="GENE",
                )
            )

    print("Finished graph creation")
    print(f"Genes:      {len(gene_cache):,}")
    print(f"Proteins:   {len(protein_cache):,}")
    print(f"Cell types: {len(celltype_cache):,}")

"""


