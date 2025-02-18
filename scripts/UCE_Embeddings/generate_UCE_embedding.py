import os
import argparse
import torch
import scanpy as sc
from omnicell.data.catalogue import Catalogue
from omnicell.data.loader import DataLoader, DatasetDetails
from omnicell.constants import DATA_CATALOGUE_PATH


EMBEDDING_PATH = '/orcd/data/omarabu/001/Omnicell_datasets/unprocessed_embeddings/ESM2_UCE_HomoSapiens.pt'
EMBEDDING_NAME = 'ESM2_UCE_HomoSapiens'

def main():
    parser = argparse.ArgumentParser(description='Fetch UCE ESM embeddings for perturbations')
    parser.add_argument('--dataset_name', type=str, required=True, 
                       help='Name of the dataset')



    args = parser.parse_args()

    # Load dataset details
    ds_details = Catalogue.get_dataset_details(args.dataset_name)

    # Check if embedding already exists
    if EMBEDDING_NAME in ds_details.pert_embeddings:
        print(f"Embedding {EMBEDDING_NAME} already exists for dataset {args.dataset_name}")
        return

    # Load UCE ESM embeddings
    uce_embeddings = torch.load(EMBEDDING_PATH)
    
    # Load dataset
    adata = sc.read(ds_details.path, backed='r')
    pert_key = ds_details.pert_key
    control = ds_details.control
    
    # Get perturbations (excluding control)
    dataset_perts = [p for p in adata.obs[pert_key].unique() if p != control]
    
    # Match embeddings with dataset perturbations
    embeddings = []
    found_perts = []
    missing_count = 0
    
    for pert in dataset_perts:
        if pert in uce_embeddings:
            embeddings.append(uce_embeddings[pert])
            found_perts.append(pert)
        else:
            print(f"Warning: {pert} not found in UCE embeddings")
            missing_count += 1
    
    if missing_count > 0:
        print(f"Total missing perturbations: {missing_count}/{len(dataset_perts)}")
    
    if not embeddings:
        raise ValueError("No matching perturbations found in UCE embeddings")
    
    # Convert to tensor
    embeddings = torch.stack(embeddings)
    
    # Save embeddings
    save_dir = f"{ds_details.folder_path}/pert_embeddings/"
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        "embedding": embeddings,
        "pert_names": found_perts
    }, f"{save_dir}/{EMBEDDING_NAME}.pt")
    
    # Update catalogue
    Catalogue.register_new_pert_embedding(args.dataset_name, EMBEDDING_NAME)
    print(f"Successfully saved {len(found_perts)} embeddings in {EMBEDDING_NAME} for {args.dataset_name}")

if __name__ == '__main__':
    main()