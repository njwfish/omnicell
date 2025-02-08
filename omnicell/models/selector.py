import logging
from omnicell.data.loader import DataLoader
from typing import Dict
import numpy as np
from omnicell.config.config import Config, ModelConfig


logger = logging.getLogger(__name__)

def load_model(model_config: ModelConfig, loader: DataLoader, pert_embedding: Dict[str, np.ndarray], input_dim: int, device, pert_ids):
    
    model_name = model_config.name
    model_parameters = model_config.parameters

    if "nearest-neighbor_pert_emb" in model_name:
        from omnicell.models.nearest_neighbor.predictor import NearestNeighborPredictor
        logger.info("Nearest Neighbor model selected")
        model = NearestNeighborPredictor(model_parameters, device)

    elif 'nearest-neighbor_gene_dist' in model_name:
        from omnicell.models.nearest_neighbor.gene_distance import NearestNeighborPredictor
        logger.info("Nearest Neighbor Gene Distance model selected")
        model = NearestNeighborPredictor(model_parameters)

    elif 'flow' in model_name:
        from omnicell.models.flows.flow_predictor import FlowPredictor
        logger.info("Flow model selected")
        model = FlowPredictor(model_parameters, input_dim, pert_embedding)

    elif 'llm' in model_name:
        from omnicell.models.llm.llm_predictor import LLMPredictor
        logger.info("Transformer model selected")
        model = LLMPredictor(model_parameters, input_dim, device, pert_ids)
        
    elif 'vae' in model_name:
        from omnicell.models.VAE.vae import VAE
        logger.info("VAE model selected")
        model = VAE(model_parameters, input_dim, device, pert_ids)
    
    elif 'cell_emb' in model_name:
        from omnicell.models.cell_emb.cell_emb_predictor import CellEmbPredictor
        logger.info("Cell Emb model selected")
        model = CellEmbPredictor(model_parameters, input_dim)

    elif 'scVIDR' in model_name:
        from omnicell.models.VAE.scVIDR_predictor import ScVIDRPredictor
        logger.info("scVIDR model selected")
        model = ScVIDRPredictor(model_parameters, input_dim, device, pert_ids)

    elif "test" in model_name:
        from omnicell.models.dummy_predictors.perfect_predictor import PerfectPredictor
        logger.info("Test model selected")
        adata_cheat = loader.get_complete_training_dataset()
        model = PerfectPredictor(adata_cheat)
    elif "nn_oracle" in model_name:
        from omnicell.models.dummy_predictors.oracle_nearest_neighbor import OracleNNPredictor
        logger.info("NN Oracle model selected")
        adata_cheat = loader.get_complete_training_dataset()
        model = OracleNNPredictor(adata_cheat, model_parameters)

    elif "sclambda" in model_name:
        from omnicell.models.sclambda.model import ModelPredictor
        logger.info("SCLambda model selected")
        model = ModelPredictor(input_dim, device, pert_embedding, **model_parameters)

    elif "mean_model" in model_name:
        from omnicell.models.mean_models.model import MeanPredictor
        logger.info("Mean model selected")
        model = MeanPredictor(model_parameters, pert_embedding)
        
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
        model = SCOT(adata_cheat, pert_embedding, **model_parameters)

    elif "gears" in model_name:
        from omnicell.models.gears.predictor import GEARSPredictor
        logger.info("GEARS model selected")
        model = GEARSPredictor(device, model_parameters)
        
    else:
        raise ValueError(f'Unknown model name {model_name}')
    
    return model
