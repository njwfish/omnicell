import scanpy as sc
import numpy as np
import argparse
from omnicell.data.catalogue import Catalogue
import yaml
import os
import json
import pandas as pd


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



#Do we handle the across

def main():
    parser = argparse.ArgumentParser(description='Analysis settings.')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument('--split_mode', type=str, default='ood', help='Mode of the split config')
    parser.add_argument('--split_size', type=int, help='Size of the split, # of perts for evaluation')
    parser.add_argument('--number_splits', type=int, help='Number of splits')
    parser.add_argument('--target_cell', type=str, default='MAX', help='Target cell for the split, if MAX then the cell with the most observations is selected')
    parser.add_argument('--most_perturbative', type=float, default=None, help='If set to a fraction then the heldout perts are sampled from the most perturbative perts')


    args = parser.parse_args()






    ds_details = Catalogue.get_dataset_details(args.dataset)
    
    ds_path = ds_details.path

    print(f"Loading dataset from {ds_path}")
    adata = sc.read(ds_path)
    

    target_cell = args.target_cell

    #Select the cell type with the most observations
    if target_cell == "MAX":
        target_cell = adata.obs[ds_details.cell_key].value_counts().idxmax()

    else:
        target_cell = str(target_cell)

    
    perts_in_ds = [x for x in adata.obs[ds_details.pert_key].unique() if x != ds_details.control]

    print(f"Loaded dataset with {len(perts_in_ds)} non control perts")
    

    split_name = f"rs_accP_{target_cell}_{args.split_mode}_ss:ns_{args.split_size}_{args.number_splits}"
    if args.most_perturbative is not None:
        assert ds_details.precomputed_DEGs is not None, "DEGs must be computed for this option"
        assert 0 < args.most_perturbative < 1, "most_perturbative must be a fraction between 0 and 1"

        split_name += f"_most_pert_{args.most_perturbative}"


    


    splits_path = f"configs/splits/{args.dataset}/random_splits/{split_name}"



    print(f"Selected cell {target_cell}")

    for i in range(args.number_splits):

        #Select a random subset of perturbations
        candidate_perts = None
        if args.most_perturbative is not None:
            DEGs_path = f"{ds_details.folder_path}/DEGs.json"
            DEGs_all = json.load(open(DEGs_path, 'r'))


            perts_in_cell_type = sorted(list(DEGs_all[target_cell].keys()))

            DEGs_target = {}
            for pert in DEGs_all[target_cell]:
                df = pd.DataFrame.from_dict(DEGs_all[target_cell][pert], orient='index')
                df = df[df['pvals_adj'] < 0.05]
                DEGs_target[pert] = df

            
            number_DEGs_per_pert = {}
            for pert in perts_in_cell_type:
                if pert in DEGs_target:
                    number_DEGs_per_pert[pert] = len(DEGs_target[pert])
                else:
                    number_DEGs_per_pert[pert] = 0

            #Select the most perturbative perts based on the number of DEGs
            number_perts = int(len(perts_in_cell_type) * args.most_perturbative)
            candidate_perts = sorted(number_DEGs_per_pert, key=number_DEGs_per_pert.get, reverse=True)[:number_perts]
        
        else:
            candidate_perts = perts_in_ds


        heldout_perts = [str(x) for x in np.random.choice(candidate_perts, args.split_size, replace=False)]

        #Create the split config

        #Separating split_name and split number with hyphen ensures the splits will be grouped together (and only together) in the results directory
        split_config = template_split_config(f"{split_name}-split_{i}", args.split_mode, args.dataset, [], heldout_perts)


        eval_config = template_eval_config(f"{split_name}-eval_{i}", args.dataset, [[str(target_cell), str(pert)] for pert in heldout_perts])

        split_path = f"{splits_path}/split_{i}"

        #make the directory
        os.makedirs(split_path, exist_ok=True)

        print(f"Saving split {i} to {split_path}")
        
        #Save the split config
        with open(f"{split_path}/split_config.yaml", 'w+') as f:
            yaml.dump(split_config, f)
        
        #Save the eval config
        with open(f"{split_path}/eval_config.yaml", 'w+') as f:
            yaml.dump(eval_config, f)





if __name__ == "__main__":
    main()