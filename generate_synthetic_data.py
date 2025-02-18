import argparse
import scanpy as sc
import yaml
from pathlib import Path
import logging
import numpy as np
import datetime
from omnicell.constants import GENE_EMBEDDING_KEY, PERT_KEY, CELL_KEY, CONTROL_PERT, DATA_CATALOGUE_PATH
from omnicell.data.utils import prediction_filename
from omnicell.processing.utils import to_dense, to_coo
from omnicell.config.config import Config
from omnicell.data.loader import DataLoader
from omnicell.data.catalogue import Catalogue
import torch
import pickle
import os
from omnicell.models.utils.datamodules import get_dataloader
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
    parser.add_argument('--slurm_id', type=int, default=1, help='Slurm id for the job')
    parser.add_argument('--slurm_array_task_id', type=int, default=1, help='Slurm array task id')
    parser.add_argument('-l', '--log', dest='loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       help="Set the logging level", default='INFO')
    
    args = parser.parse_args()

    # Configure logging
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    config = Config.from_yamls(model_yaml=args.model_config,
                             etl_yaml=args.etl_config,
                             datasplit_yaml=args.datasplit_config)
    
    logfile_name = f'synthetic_data_{args.slurm_id}_{args.slurm_array_task_id}_{config.model_config.name}_{config.etl_config.name}.log'
    
    logging.basicConfig(
        filename=logfile_name,
        filemode='w',
        level=args.loglevel,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.getLogger('numba').setLevel(logging.CRITICAL)
    logging.getLogger('pytorch_lightning').setLevel(logging.CRITICAL)
    
    logger.info("Starting synthetic data generation process")
    
    synthetic_data_name = config.get_synthetic_config_ID()
    dataset_details = Catalogue.get_dataset_details(args.dataset)
    
    if synthetic_data_name in dataset_details.synthetic_versions:
        logger.info(f"Synthetic data {synthetic_data_name} already exists for dataset {args.dataset}, skipping.")
        return
        
    logger.info(f"Loading data for dataset {args.dataset}")
    loader = DataLoader(config)
    adata, pert_embedding = loader.get_training_data()
    
    input_dim = adata.X.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pert_ids = adata.obs[PERT_KEY].unique()
    

    
    model = load_model(config.model_config, loader, pert_embedding, input_dim, device, pert_ids)
    
    datapath = f"{dataset_details.folder_path}/synthetic_data/{synthetic_data_name}"
    os.makedirs(datapath, exist_ok=True)
    
    logger.info(f"Generating synthetic data to {datapath}")
    for i, data_dict in enumerate(model.generate_synthetic(adata=adata)):
        logger.debug(f"Saving batch {i} of synthetic data")
        with open(f'{datapath}/synthetic_counterfactuals_{i}.pkl', 'wb') as f:
            logger.debug(f"Saving batch {i} of synthetic")
            pickle.dump(data_dict, f)
    
    logger.info("Saving configuration")
    with open(f'{datapath}/config.yaml', 'w') as f:
        yaml.dump(config.to_dict(), f)
    
    Catalogue.register_new_synthetic_version(args.dataset, config.get_synthetic_config_ID())
    logger.info("Synthetic data generation completed")
    
    # Move log file to results directory
    os.rename(logfile_name, f"{datapath}/{logfile_name}")

if __name__ == "__main__":
    main()