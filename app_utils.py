_UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
_PUBCHEM_COMPOUND = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/JSON"
from gem_core import Gem


_TISSLIST_URL = "https://www.uniprot.org/docs/tisslist.txt"
_UBERON_URL = "http://purl.obolibrary.org/obo/uberon/basic.obo"

_HTTP_TIMEOUT = 30
_UNIPROT_FIELDS = "accession,protein_name,gene_names,cc_function,go,cc_disease,cc_tissue_specificity"
_MAX_PROTEINS_PER_ORGAN = 50

BRAIN_TERMS = [
    "Brain",
    "CNS",
    "Nervous system",
    "Limbic system",
    "Brain cortex",
    "Cerebellum",
    "Hippocampus",
    "Hypothalamus",
    "Brain stem",
    "Spinal cord",
    "Ganglion",
    "Glial cell",
    "Neuron",
    "Microglia",
    "Astrocyte",
]


gem = Gem()
QUERY_TRANSFORM_PROMPT=f"""
You are a query tansformator. Analyze the given Prompt and create a list of 5 comma sepparated keyywords that 
describe this biological process.  
Return just the keywords comma separated, no additional text.
"""