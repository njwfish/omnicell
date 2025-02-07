
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
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
print(torch.cuda.is_available())





def main():

    parser = argparse.ArgumentParser(description='Generate llm embeddings for perts')

    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')

    parser.add_argument('--model_name', choices=["MMedllama-3-8B", "llamaPMC-13B", "llamaPMC-7B", "bioBERT"], help='Name of the model to use for embedding generation')

    args = parser.parse_args()

    assert args.dataset_name is not None, "Please provide a dataset name"


    ds_details = Catalogue.get_dataset_details(args.dataset_name)

    if args.model_name in ds_details.pert_embeddings:
        print(f"Embedding {args.model_name} already exists for dataset {args.dataset_name} - Terminating")
        print(f"Available embeddings: {ds_details.pert_embeddings}")
        return

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")

    model = None
    tokenizer = None

    if args.model_name == "MMedllama-3-8B":
        tokenizer = AutoTokenizer.from_pretrained("Henrychur/MMed-Llama-3-8B")
        model = AutoModelForCausalLM.from_pretrained("Henrychur/MMed-Llama-3-8B", torch_dtype=torch.float16).to(device)
    elif args.model_name == "llamaPMC-13B":
        tokenizer = transformers.LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
        model = transformers.LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B').to(device)
    elif args.model_name == "llamaPMC-7B":
        tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
        model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B').to(device)
    elif args.model_name == "bioBERT":
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1").to(device)


   

    print(f"Loading dataset from {ds_details.path}")
    adata = sc.read(ds_details.path, backed='r') 


    pert_names = [x for x in adata.obs[ds_details.pert_key].unique() if x != ds_details.control]


    tokenizer.pad_token = tokenizer.eos_token

    

    embeddings = []

    for _, g in enumerate(pert_names):

        inputs = tokenizer(g, return_tensors="pt").to("cuda")


        outputs = model(**inputs)
        


        if args.model_name != "bioBERT":
            outputs = outputs.logits.squeeze(0)
            outputs = outputs.mean(axis=0)
        else:
            outputs = torch.squeeze(outputs.pooler_output)        

        embeddings.append(outputs.cpu().detach())
        

    embeddings = torch.stack(embeddings)

    print(f"Embeddings shape: {embeddings.shape}")

    save_path_pert_emb = f"{ds_details.folder_path}/pert_embeddings/"

    os.makedirs(save_path_pert_emb, exist_ok=True)
    
    
    torch.save({"embedding": embeddings, "pert_names" : list(pert_names)},f"{save_path_pert_emb}/{args.model_name}.pt")



    #Register the new embedding in the catalogue, This modifies the underlying yaml file
    Catalogue.register_new_pert_embedding(args.dataset_name, args.model_name)

    print("Pert embedding generated and saved successfully")



if __name__ == '__main__':
    main()