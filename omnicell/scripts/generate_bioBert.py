
from regex import E
import scanpy as sc
from omnicell.data.loader import DataLoader, DatasetDetails
import torch 
from transformers import AutoTokenizer, AutoModel
import argparse
from omnicell.constants import DATA_CATALOGUE_PATH
import json
from omnicell.data.catalogue import Catalogue, DatasetDetails

import logging

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
    with open(ds_details.path, 'rb') as f:
        adata = sc.read_h5ad(f)


    

    perts = [x for x in adata.obs[pert_key].unique() if x != control_pert]

    print(f"Loaded dataset with {len(perts)} non control perts")



    print(f"Fetching model from huggingface")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

    # Load model
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")


    print(f"Generating embeddings for {len(perts)} perts")
    embeddings = {}

    for pert in perts:
        text = pert.upper()

        inputs = tokenizer(text, return_tensors="pt")

        # Get the embeddings
        outputs = model(**inputs)

        print(f"Outputs keys {outputs.keys()}")

        embedding = torch.squeeze(outputs.pooler_output)

        embedding = embedding.detach().cpu()
        embedding.requires_grad = False

        embeddings[pert] = embedding

        raise Exception("Stop here")


    #Overwrites any existing file with the same name
    torch.save(embeddings, f"{ds_details.folder_path}/BioBERT.pt")

    print(f"Size of the embedding: {len(embeddings)}")


    #Register the new embedding in the catalogue, This modifies the underlying yaml file
    catalogue.register_new_pert_embedding(args.dataset_name, "BioBERT")




if __name__ == '__main__':
    main()