
import argparse
import scanpy as sc
import yaml
from pathlib import Path
import logging
import numpy as np

from omnicell.constants import GENE_EMBEDDING_KEY, PERT_KEY, CELL_KEY, CONTROL_PERT, DATA_CATALOGUE_PATH
from omnicell.data.utils import prediction_filename
from omnicell.processing.utils import to_dense, to_coo
from omnicell.config.config import Config
from omnicell.data.loader import DataLoader
from omnicell.data.catalogue import Catalogue

import torch
import pickle
from omnicell.models.utils.datamodules import get_dataloader
import logging 

from omnicell.models.selector import load_model

from omnicell.models.scot.sampling_utils import generate_batched_counterfactuals
logger = logging.getLogger(__name__)



def main():
    print("Starting synthetic data generation...")
    parser = argparse.ArgumentParser(description='Analysis settings.')
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument('--model_config', type=str, default='ood', help='Mode of the split config')
    parser.add_argument('--datasplit_config', type=str, default=None, help='Path to yaml config of the datasplit.')
    parser.add_argument('--etl_config', type=str, default=None, help='Path to yaml config file of the etl process.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')

    args = parser.parse_args()

    batch_size = args.batch_size
    config = Config.from_yamls(model_yaml = args.model_config,
                               etl_yaml   = args.etl_config, 
                               datasplit_yaml = args.datasplit_config)

    synthetic_data_name = config.get_synthetic_config_ID()
    dataset_details = Catalogue.get_dataset_details(args.dataset)

    if synthetic_data_name in dataset_details.synthetic_versions:
        print(f"Synthetic data with name {synthetic_data_name} already exists for dataset {args.dataset}, skipping.")
        return



    # Initialize data loader and load training data
    loader = DataLoader(config)
    adata, pert_rep_map = loader.get_training_data()

    adata, pert_embedding = loader.get_training_data()
        
    input_dim = adata.obsm['embedding'].shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pert_ids = adata.obs[PERT_KEY].unique()


    model = load_model(config.model_config, loader, pert_embedding, input_dim, device, pert_ids)

    dset, dl = get_dataloader(adata, pert_ids=np.array(adata.obs[PERT_KEY].values), offline=False, pert_map=pert_rep_map, collate='ot')


    datapath = f"{dataset_details.folder_path}/synthetic_data/{synthetic_data_name}"


    #We generate the synthetic data
    for i, data_dict in enumerate(generate_batched_counterfactuals(model, dset)):
        with open(f'{datapath}/synthetic_counterfactuals_{i}.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
    
    
    #We dump the config
    with open(f'{datapath}/config.yaml', 'w') as f:
        yaml.dump(config.to_dict(), f)

    Catalogue.register_new_synthetic_version(args.dataset, config.get_synthetic_config_ID())
        
    
if __name__ == "__main__":
    main()
                            



