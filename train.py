import argparse
import scanpy as sc
import yaml
from pathlib import Path
import sys
import os
import hashlib
import json
import scipy
import logging
import numpy as np
import random
from omnicell.constants import GENE_EMBEDDING_KEY, PERT_KEY, CELL_KEY, CONTROL_PERT, DATA_CATALOGUE_PATH
from omnicell.data.utils import prediction_filename
from omnicell.processing.utils import to_dense, to_coo
from omnicell.config.config import Config
from omnicell.data.loader import DataLoader
from omnicell.data.catalogue import Catalogue
import time
import datetime
from omnicell.models.selector import load_model
import torch


logger = logging.getLogger(__name__)


random.seed(42)


def main(*args):
    print("Running main")
    parser = argparse.ArgumentParser(description='Analysis settings.')


    parser.add_argument('--datasplit_config', type=str, default=None, help='Path to yaml config of the datasplit.')
    parser.add_argument('--embedding_config', type=str, default=None, help='Path to yaml config file of the embeddings.')
    parser.add_argument('--etl_config', type=str, default=None, help='Path to yaml config file of the etl process.')
    parser.add_argument('--model_config', type=str, default=None, help='Path to yaml config file of the model.')
    parser.add_argument('--eval_config', type=str, default=None, help='Path to yaml config file of the evaluations, if none provided the model will only be trained.')
    parser.add_argument('--test_mode', action='store_true', default=False, help='Run in test mode, datasetsize will be capped at 10000')
    parser.add_argument('--slurm_id', type=int, default=1, help='Slurm id for the job, useful for arrays')
    parser.add_argument('--slurm_array_task_id', type=int, default=1, help='Slurm array task id, useful for arrays')
    parser.add_argument('-l', '--log', dest='loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level (default: %(default)s)", default='INFO')

    args = parser.parse_args()

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H:%M:%S")

    config = Config.from_yamls(model_yaml = args.model_config,
                               etl_yaml   = args.etl_config, 
                               datasplit_yaml = args.datasplit_config,
                               embed_yaml = args.embedding_config,
                               eval_yaml  = args.eval_config)

    logfile_name = f'output_{args.slurm_id}_{args.slurm_array_task_id}_{config.model_config.name}_{config.etl_config.name}_{config.datasplit_config.name}.log'

    logging.basicConfig(
        filename=logfile_name, 
        filemode= 'w', level=args.loglevel, format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    #This is polluting the output
    logging.getLogger('numba').setLevel(logging.CRITICAL)
    logging.getLogger('pytorch_lightning').setLevel(logging.CRITICAL)
    
    logger.info("Application started")

    loader = DataLoader(config)
    
    adata, pert_embedding = loader.get_training_data()
        
    input_dim = adata.X.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pert_ids = adata.obs[PERT_KEY].unique()


    logger.info(f"Data loaded, # of cells: {adata.shape[0]}, # of features: {input_dim} # of perts: {len(pert_ids)}")
    logger.debug(f"Number of control cells {len(adata[adata.obs[PERT_KEY] == CONTROL_PERT])}")
    logger.info(f"Running experiment on {device}")


    model = load_model(config.model_config, loader, pert_embedding, input_dim, device, pert_ids)

    model_savepath = Path(f"{config.get_train_path()}/training")

    if hasattr(model, 'save') and hasattr(model, 'load'):
        # Path depends on hash of config
        if model.load(model_savepath):
            logger.info(f"Model already trained, loaded model from {model_savepath}")
        
        else:
            logger.info("Model not trained, training model")
            model.train(adata, model_savepath)
            logger.info(f"Training completed, saving model to {model_savepath}")
            os.makedirs(model_savepath, exist_ok=True)

            model.save(model_savepath)
            
            # We only save the training config (ETL + Split + Model)
            logger.info(f"Saving Training config to {model_savepath}")
            with open(f"{model_savepath}/training_config.yaml", 'w+') as f:
                yaml.dump(config.get_training_config().to_dict(), f, indent=2, default_flow_style=False)
    else:
        logger.info("Model does not support saving/loading, training from scratch")
        model.train(adata, model_savepath)
        logger.info("Training completed")    

    # If we have an evaluation config, we are going to evaluate
    if args.eval_config is not None and hasattr(model, 'make_predict'):
        logger.info("Running evaluation")
        
        results_path = config.get_eval_path()
        logger.info(f"Will save results to {results_path}")

        #Saving run config
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        with open(f"{results_path}/config.yaml", 'w+') as f:
            yaml.dump(config.to_dict(), f, indent=2, default_flow_style=False)

        # evaluate each pair of cells and perts
        for cell_id, pert_id, ctrl_data, gt_data in loader.get_eval_data():
            logger.debug(f"Making predictions for cell: {cell_id}, pert: {pert_id}")


            preds = model.make_predict(ctrl_data, pert_id, cell_id)

            preds = preds
            control = ctrl_data.X
            ground_truth = gt_data.X

            
            #No log1p in the config --> we need to log normalize the results before saving them for evals to work
            if not config.etl_config.log1p:
                preds = np.log1p(preds)
                control = np.log1p(control)
                ground_truth = np.log1p(ground_truth)
         
            preds = to_coo(preds)
            control  = to_coo(control)
            ground_truth = to_coo(ground_truth)



            #TODO: We only need to save one control file per cell, if we have several perts we can reuse the same control file
            scipy.sparse.save_npz(f"{results_path}/{prediction_filename(pert_id, cell_id)}-preds", preds)
            scipy.sparse.save_npz(f"{results_path}/{prediction_filename(pert_id, cell_id)}-control", control)
            scipy.sparse.save_npz(f"{results_path}/{prediction_filename(pert_id, cell_id)}-ground_truth", ground_truth)

        logger.info("Evaluation completed")
        logger.info("Saving logfile to results folder")

        os.rename(logfile_name, f"{results_path}/{logfile_name}")


if __name__ == '__main__':
    main()
