from typing import List

from extract_data.functions.file.check_ds_exist import check_ds_exists

# biomedical corpora.
def handle_uniprot():
    r"""
    Whole Human KB was manually extracted and can be find here:
    C:\Users\wired\OneDrive\Desktop\Projects\aws_to_bucket\extract_data\filtered_data\checkpoints\uniprotkb_AND_model_organism_9606_2024_12_29.json

    Or alternatively downloaded here:
    https://www.uniprot.org/uniprotkb?query=*&facets=model_organism%3A9606
    """
    print("UniProt request...")
    path = r"C:\Users\wired\OneDrive\Desktop\Projects\aws_to_bucket\extract_data\filtered_data\checkpoints\uniprotkb_9606.json"
    uniprot_9606_content: List = check_ds_exists(path)
    for item in uniprot_9606_content:
        continue




















