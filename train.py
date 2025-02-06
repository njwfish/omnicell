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
import torch


logger = logging.getLogger(__name__)


random.seed(42)

def get_model(model_name, config_model, loader, pert_embedding, input_dim, device, pert_ids):
    
    if pert_embedding is not None:
        pert_keys = list(pert_embedding.keys())
        pert_rep = np.array([pert_embedding[k] for k in pert_keys])
        pert_map = {k: i for i, k in enumerate(pert_keys)}
    else:
        pert_rep = None
        pert_map = None


    if "nearest-neighbor_pert_emb" in model_name:
        from omnicell.models.nearest_neighbor.predictor import NearestNeighborPredictor
        logger.info("Nearest Neighbor model selected")
        model = NearestNeighborPredictor(config_model, device, pert_rep=pert_rep, pert_map=pert_map)

    elif 'nearest-neighbor_gene_dist' in model_name:
        from omnicell.models.nearest_neighbor.gene_distance import NearestNeighborPredictor
        logger.info("Nearest Neighbor Gene Distance model selected")
        model = NearestNeighborPredictor(config_model)

    elif 'flow' in model_name:
        from omnicell.models.flows.flow_predictor import FlowPredictor
        logger.info("Flow model selected")
        model = FlowPredictor(config_model, input_dim, pert_rep, pert_map)

    elif 'llm' in model_name:
        from omnicell.models.llm.llm_predictor import LLMPredictor
        logger.info("Transformer model selected")
        model = LLMPredictor(config_model, input_dim, device, pert_ids)
        
    elif 'vae' in model_name:
        from omnicell.models.VAE.vae import VAE
        logger.info("VAE model selected")
        model = VAE(config_model, input_dim, device, pert_ids)
    
    elif 'cell_emb' in model_name:
        from omnicell.models.cell_emb.cell_emb_predictor import CellEmbPredictor
        logger.info("Cell Emb model selected")
        model = CellEmbPredictor(config_model, input_dim, device, pert_ids)

    elif 'scVIDR' in model_name:
        from omnicell.models.VAE.scVIDR_predictor import ScVIDRPredictor
        logger.info("scVIDR model selected")
        model = ScVIDRPredictor(config_model, input_dim, device, pert_ids)

    elif "test" in model_name:
        from omnicell.models.dummy_predictors.perfect_predictor import PerfectPredictor
        logger.info("Test model selected")
        adata_cheat = loader.get_complete_training_dataset()
        model = PerfectPredictor(adata_cheat)
    elif "nn_oracle" in model_name:
        from omnicell.models.dummy_predictors.oracle_nearest_neighbor import OracleNNPredictor
        logger.info("NN Oracle model selected")
        adata_cheat = loader.get_complete_training_dataset()
        model = OracleNNPredictor(adata_cheat, config_model)

    elif "sclambda" in model_name:
        from omnicell.models.sclambda.model import ModelPredictor
        logger.info("SCLambda model selected")
        model = ModelPredictor(input_dim, device, pert_embedding, **config_model)

    elif "mean_model" in model_name:
        from omnicell.models.mean_models.model import MeanPredictor
        logger.info("Mean model selected")
        model = MeanPredictor(config_model, pert_embedding)
        
    elif "control_predictor" in model_name:
        from omnicell.models.dummy_predictors.control_predictor import ControlPredictor
        logger.info("Control model selected")
        adata_cheat = loader.get_complete_training_dataset()
        model = ControlPredictor(adata_cheat)
    
    elif "proportional_scot" in model_name:
        from omnicell.models.scot.proportional import ProportionalSCOT
        logger.info("Proportional SCOT model selected")
        adata_cheat = loader.get_complete_training_dataset()
        model = ProportionalSCOT(adata_cheat)

    elif "scot" in model_name:
        from omnicell.models.scot.scot import SCOT
        logger.info("SCOT model selected")
        adata_cheat = loader.get_complete_training_dataset()
        model = SCOT(adata_cheat, **config_model)
    #MMB
    elif "cinemaot" in model_name:
        from omnicell.models.cinemaot_model import CinemaOTModel
        logger.info("CINEMA-OT model selected")
        model = CinemaOTModel(config_model) 
        #model = CinemaOTModel(config_model,pert_key=config_model.get("pert_key", "perturbation"), control=config_model.get("control", "No stimulation"),  device=device)
    elif "gears" in model_name:
        from omnicell.models.gears.predictor import GEARSPredictor
        logger.info("GEARS model selected")
        model = GEARSPredictor(device, config_model)
        
    else:
        raise ValueError(f'Unknown model name {model_name}')
    
    return model

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
        
    input_dim = adata.obsm['embedding'].shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pert_ids = adata.obs[PERT_KEY].unique()

    gene_emb_dim = adata.varm[GENE_EMBEDDING_KEY].shape[1] if GENE_EMBEDDING_KEY in adata.varm else None

    logger.info(f"Data loaded, # of cells: {adata.shape[0]}, # of features: {input_dim} # of perts: {len(pert_ids)}")
    logger.debug(f"Number of control cells {len(adata[adata.obs[PERT_KEY] == CONTROL_PERT])}")
    logger.info(f"Running experiment on {device}")

    logger.debug(f"Training data loaded, perts are: {adata.obs[PERT_KEY].unique()}")

    model = get_model(config.model_config.name, config.model_config.parameters, loader, pert_embedding, input_dim, device, pert_ids)

    model_savepath = f"{config.get_train_path()}/training"

    if hasattr(model, 'save') and hasattr(model, 'load'):
        # Path depends on hash of config
        if model.load(model_savepath):
            logger.info(f"Model already trained, loaded model from {model_savepath}")
        
        else:
            logger.info("Model not trained, training model")
            model.train(adata)
            logger.info(f"Training completed, saving model to {model_savepath}")
            os.makedirs(model_savepath, exist_ok=True)

            model.save(model_savepath)
            
            # We only save the training config (ETL + Split + Model)
            logger.info(f"Saving Training config to {model_savepath}")
            with open(f"{model_savepath}/training_config.yaml", 'w+') as f:
                yaml.dump(config.get_training_config().to_dict(), f, indent=2, default_flow_style=False)
    else:
        logger.info("Model does not support saving/loading, training from scratch")
        if "cinemaot" in config.model_config.name:
            sc.pp.pca(adata)  # CINEMA-OT requires AnnData with PCA
            model.train(adata)
        else:
            model.train(adata)
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


            preds = model.make_predict(ctrl_data, pert_id, cell_id) #might need to do preds.X for CINEMAOT

            preds = preds
            control = ctrl_data.X
            ground_truth = gt_data.X

            
            #No log1p in the config --> we need to log normalize the results before saving them for evals to work
            if not config.etl_config.log1p:
                preds = np.log1p(preds)
                control = np.log1p(control)
                ground_truth = np.log1p(ground_truth)
         
            preds = to_coo(preds)
            control  = to_coo(ctrl_data.X)
            ground_truth = to_coo(gt_data.X)



            #TODO: We only need to save one control file per cell, if we have several perts we can reuse the same control file
            scipy.sparse.save_npz(f"{results_path}/{prediction_filename(pert_id, cell_id)}-preds", preds)
            scipy.sparse.save_npz(f"{results_path}/{prediction_filename(pert_id, cell_id)}-control", control)
            scipy.sparse.save_npz(f"{results_path}/{prediction_filename(pert_id, cell_id)}-ground_truth", ground_truth)

        logger.info("Evaluation completed")
        logger.info("Saving logfile to results folder")

        os.rename(logfile_name, f"{results_path}/{logfile_name}")


if __name__ == '__main__':
    main()
