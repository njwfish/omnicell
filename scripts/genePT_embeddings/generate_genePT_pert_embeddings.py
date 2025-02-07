
from turtle import back
from matplotlib import axis
import scanpy as sc
from zmq import device
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
import pickle
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM



EMBEDDING_NAME = 'GenePT'



def main():

    parser = argparse.ArgumentParser(description='Generate llm embeddings for perts')

    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')

    args = parser.parse_args()

    assert args.dataset_name is not None, "Please provide a dataset name"


    ds_details = Catalogue.get_dataset_details(args.dataset_name)

    if EMBEDDING_NAME in ds_details.pert_embeddings:
        print(f"Embedding {EMBEDDING_NAME} already exists for dataset {args.dataset_name} - Terminating")
        return




    DATA_DIR = '/orcd/data/omarabu/001/Omnicell_datasets/GenePT_emebdding_v2'

    with open(os.path.join(DATA_DIR, 'GenePT_gene_protein_embedding_model_3_text.pickle.'), 'rb') as f:
        pert_embeddings = pickle.load(f)


    print(f"Loading dataset from {ds_details.path}")

    with open(ds_details.path, 'rb') as f:
        adata = sc.read_h5ad(f)




    pert_names = [x for x in adata.obs[ds_details.pert_key].unique() if x != ds_details.control]


    embeddings = []
    names = []

    for i, g in enumerate(pert_names):
        if g in pert_embeddings:

            p_emb = pert_embeddings[g]
            p_emb = torch.tensor(p_emb)

            embeddings.append(p_emb)
            names.append(g)

        

    embeddings = torch.stack(embeddings)

    print(f"Embeddings shape: {embeddings.shape}")

    save_path_gene_emb = f"{ds_details.folder_path}/pert_embeddings/"

    os.makedirs(save_path_gene_emb, exist_ok=True)
    
    
    torch.save({"embedding": embeddings, "pert_names" : names},f"{save_path_gene_emb}/{EMBEDDING_NAME}.pt")


    #Register the new embedding in the catalogue, This modifies the underlying yaml file
    Catalogue.register_new_pert_embedding(args.dataset_name, EMBEDDING_NAME)

    print("Pert embedding generated and saved successfully")



if __name__ == '__main__':
    main()