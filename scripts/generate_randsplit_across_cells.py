import scanpy as sc
import numpy as np
import argparse
from omnicell.data.catalogue import Catalogue
import yaml
import os


def template_split_config(name, mode, dataset_name, holdout_cells, holdout_perts) -> str:
   return {
        "name": name,
        "mode": mode,
        "dataset": dataset_name,
        "holdout_cells": holdout_cells,
        "holdout_perts": holdout_perts,
        }

def template_eval_config(name, dataset, evaluation_targets) -> str:
   
   return {
        "name": name,
        "dataset": dataset,
        "evaluation_targets": evaluation_targets,
        }




def main():
    parser = argparse.ArgumentParser(description='Analysis settings.')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument('--split_mode', type=str, default='ood', help='Mode of the split config')
    parser.add_argument('--split_size', type=int, help='Size of the split, # of perts for evaluation')


    args = parser.parse_args()





    catalogue = Catalogue("configs/catalogue")

    ds_details = catalogue.get_dataset_details(args.dataset)
    
    ds_path = ds_details.path

    print(f"Loading dataset from {ds_path}")
    with open(ds_path, 'rb') as f:
        adata = sc.read_h5ad(f)
    


    
    perts = [x for x in adata.obs[ds_details.pert_key].unique() if x != ds_details.control]

    print(f"Loaded dataset with {len(perts)} non control perts")
    



    path = f"configs/splits/{args.dataset}/random_splits/acrossC_{args.split_mode}_ss:{args.split_size}"



    #Getting cell types
    cell_types = adata.obs[ds_details.cell_key].unique()

    selected_pert = [str(x) for x in np.random.choice(perts, args.split_size, replace=False)]



    #Each fold we hold one cell type out
    for cell in cell_types:



        #Select a random subset of perturbations

        #Create the split config

        #Name with hyphen ensures the splits will be grouped together (and only together) in the results directory
        split_config = template_split_config(f"rs-across-cells-{args.split_mode}-split-ss:{args.split_size}_split_{cell}", args.split_mode, args.dataset, [cell], [])


        eval_config = template_eval_config(f"rs-across-cells-{args.split_mode}-split-ss:{args.split_size}_eval_{cell}", args.dataset, [[str(cell), str(pert)] for pert in selected_pert])

        path_split = f"{path}/split_{cell}"

        #make the directory
        os.makedirs(path_split, exist_ok=True)

        #Save the split config
        with open(f"{path_split}/split_config.yaml", 'w+') as f:
            yaml.dump(split_config, f)
        
        #Save the eval config
        with open(f"{path_split}/eval_config.yaml", 'w+') as f:
            yaml.dump(eval_config, f)




        #Save the configs







    #Saving the configs

    #If we holdout cells we need to make sure that we evaluate across several perts on the heldout cells

















   


if __name__ == "__main__":
    main()