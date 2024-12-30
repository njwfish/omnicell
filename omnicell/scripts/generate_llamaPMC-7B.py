
from turtle import back
import scanpy as sc
from omnicell.data.loader import DataLoader, DatasetDetails
import torch 
from transformers import AutoTokenizer, AutoModel
import argparse
from omnicell.constants import DATA_CATALOGUE_PATH
import json
from omnicell.data.catalogue import Catalogue, DatasetDetails

import logging
import scanpy as sc
from omnicell.data.loader import DataLoader, DatasetDetails
import torch 
from transformers import AutoTokenizer, AutoModel
import transformers
import numpy as np

print(torch.cuda.is_available())


logger = logging.getLogger(__name__)



def main():

    parser = argparse.ArgumentParser(description='Generate static embedding')

    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')

    args = parser.parse_args()

    assert args.dataset_name is not None, "Please provide a dataset name"




    catalogue = Catalogue(DATA_CATALOGUE_PATH)

    #Getting the dataset details from the data_catalogue.json

    ds_details = catalogue.get_dataset_details(args.dataset_name)
    pert_key = ds_details.pert_key
    control_pert = ds_details.control
    

    # Dealing with https://github.com/h5py/h5py/issues/1679
    print(f"Loading dataset from {ds_details.path}")
    adata = sc.read(ds_details.path, backed='r')

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
    'chaoyi-wu/PMC_LLAMA_7B',
    local_files_only=False
    )

    model = transformers.LlamaForCausalLM.from_pretrained(
        'chaoyi-wu/PMC_LLAMA_7B',
        local_files_only=False
    ).to("cuda")


    


    
    gene_names = adata.var["gene"]


    gene_names_idx = gene_names.index.to_numpy().astype(np.int32) - 1

    gene_names = list(gene_names)
    tokenizer.pad_token = tokenizer.eos_token

    

    for i, g in enumerate(gene_names):

        inputs = tokenizer(g, return_tensors="pt")

        outputs = model(**inputs)

        print(outputs)

        break

    
    """

    #Overwrites any existing file with the same name
    torch.save(embeddings, f"{ds_details.folder_path}/llamaPMC.pt")

    print(f"Size of the embedding: {len(embeddings)}")


    #Register the new embedding in the catalogue, This modifies the underlying yaml file
    catalogue.register_new_pert_embedding(args.dataset_name, "llamaPMC")"""




if __name__ == '__main__':
    main()