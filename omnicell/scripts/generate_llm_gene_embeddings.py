
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
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
print(torch.cuda.is_available())


logger = logging.getLogger(__name__)



def main():

    parser = argparse.ArgumentParser(description='Generate static embedding')

    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')

    parser.add_argument('--model_name', choices=["MMedllama-3-8B", "llamaPMC-13B", "llamaPMC-7B", "bioBERT"], help='Name of the model to use for embedding generation')

    args = parser.parse_args()

    assert args.dataset_name is not None, "Please provide a dataset name"


    ds_details = Catalogue.get_dataset_details(args.dataset_name)

    if args.model_name in ds_details.gene_embeddings:
        logger.info(f"Embedding {args.model_name} already exists for dataset {args.dataset_name} - Terminating")
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


    gene_names = adata.var_names


    tokenizer.pad_token = tokenizer.eos_token

    

    embeddings = []

    for i, g in enumerate(gene_names):
        if(i % 1000 == 0):
            print(f"Processed {i}/{len(gene_names)} genes")
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
    

    torch.save(embeddings, f"{ds_details.folder_path}/{args.model_name}.pt")



    #Register the new embedding in the catalogue, This modifies the underlying yaml file
    Catalogue.register_new_gene_embedding(args.dataset_name, args.model_name)

    print("Embedding generated and saved successfully")




if __name__ == '__main__':
    main()