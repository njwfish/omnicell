import scanpy as sc
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from omnicell.config.config import Config, ModelConfig
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT, PERT_EMBEDDING_KEY, SYNTHETIC_DATA_PATHS_KEY
from omnicell.data.catalogue import DatasetDetails, Catalogue
import torch
import logging
import numpy as np
import json
import pandas as pd
from pathlib import Path
import os
import hashlib

logger = logging.getLogger(__name__)


"""


                Task Definition



Predict
    \
     \           Pert 1                        Pert 2
      \
   On  \
        \
         \__________________________________________________________
         |                              |                           |
         |                              | • We see Pert 2           |
         |  Same logic as bottom right  |   at training but         |
  Cell A |  Quadrant                    |   never on cell A         |
         |                              | • We never see Pert 2     |
         |                              |   at training             |
         |                              |                           |
         |______________________________|___________________________|
         |                              |                           |
         | • We see unperturbed         | • We never see the        |
         |   Cell B at training         |   combination Cell B +    |
         | • We never see Cell B        |   Gene 2 at training but  |
  Cell B |   at training                |   we see them separately  |
         |                              | • We see Unperturbed      |
         |                              |   Cell B but never see    |
         |                              |   Gene 2 (and reverse)    |
         |                              | • We never see Pert 1     |
         |                              |   or Cell B               |
         |                              |                           |
         |______________________________|___________________________|

Bullet points are sorted by increasing difficulty


"""


#We define an enum which is either Training or Evaluation
#We can then use this to determine which data to load



#TODO: Want to include generic dataset caching, we might starting having many datasets involved in training, not just one

def get_identity_features(adata):
    perts = [p for p in np.unique(adata.obs[PERT_KEY]) if p != CONTROL_PERT]
    n_perts = len(perts)
    
    # Create identity matrix instead of using get_dummies
    identity_matrix = np.eye(n_perts, dtype=np.float32)
    
    # Create dictionary mapping perturbations to one-hot vectors
    return {pert: identity_matrix[i] for i, pert in enumerate(perts)}


class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.training_dataset_details: DatasetDetails = Catalogue.get_dataset_details(config.datasplit_config.dataset)

        logger.debug(f"Training dataset details: {self.training_dataset_details}")

        self.pert_embedding_name: Optional[str] = config.embedding_config.pert_embedding if config.embedding_config is not None else None
        self.gene_embedding_name: Optional[str] = config.embedding_config.gene_embedding if config.embedding_config is not None else None
        self.metric_space: Optional[str] = config.embedding_config.metric_space if config.embedding_config is not None else None

        
        #TODO: Handle
        self.pert_embedding_details: Optional[dict] = None

        self.eval_dataset_details: DatasetDetails = None

        #We only store the data once it has been preprocessed
        self.complete_training_adata: Optional[sc.AnnData] = None
        self.complete_eval_adata: Optional[sc.AnnData] = None
        

    # TODO: This processing should be common between the training and the eval data
    # Mutates the adata object
    def preprocess_data(self, adata: sc.AnnData, training: bool) -> sc.AnnData:

        dataset_details = self.training_dataset_details if training else self.eval_dataset_details
        dataset_name = self.config.datasplit_config.dataset if training else self.config.eval_config.dataset
        # Standardize column names and key values
        condition_key = dataset_details.pert_key
        cell_key = dataset_details.cell_key if training else self.eval_dataset_details.cell_key
        control = dataset_details.control if training else self.eval_dataset_details.control

        #TODO: If we could rename the columns it would be better

        adata.obs.rename(columns={condition_key: PERT_KEY, cell_key: CELL_KEY}, inplace=True)

        adata.obs[PERT_KEY] = adata.obs[PERT_KEY].cat.rename_categories({control: CONTROL_PERT})

        #Attaching gene embeddings
        if self.gene_embedding_name is not None:
            if self.gene_embedding_name not in dataset_details.gene_embeddings:
                raise ValueError(f"Gene Embedding {self.gene_embedding_name} is not found in gene embeddings available for dataset {dataset_name}")
            else:
                embeddings_and_gene_names = torch.load(f"{dataset_details.folder_path}/gene_embeddings/{self.gene_embedding_name}.pt")
                embedding = embeddings_and_gene_names["embedding"]
                
                gene_rep = np.array(embedding)
                adata.varm["gene_embedding"] = gene_rep


        #Getting the per embedding if it is specified
        if self.pert_embedding_name is not None:
            if self.pert_embedding_name not in self.training_dataset_details.pert_embeddings:
                raise ValueError(f"Perturbation embedding {self.pert_embedding_name} not found in embeddings available for dataset {self.training_dataset_details.name}")
            else:
                logger.info(f"Loading perturbation embedding from {self.training_dataset_details.folder_path}/{self.pert_embedding_name}.pt")
                pert_embeddings_and_name = torch.load(f"{self.training_dataset_details.folder_path}/pert_embeddings/{self.pert_embedding_name}.pt")
                embeddings = pert_embeddings_and_name["embedding"]
                pert_names = pert_embeddings_and_name["pert_names"]
                pert_embedding = {pert_names[i]: embeddings[i] for i in range(len(embeddings))}

                adata.uns[PERT_EMBEDDING_KEY] = pert_embedding

        else:
            logger.info("Using identity features for perturbations")
            pert_embedding = get_identity_features(adata)
            adata.uns[PERT_EMBEDDING_KEY] = pert_embedding


        #Getting HVG genes
        if not dataset_details.HVG and self.config.etl_config.HVG:
            logger.info(f"Filtering HVG to top 2000 genes of {adata.shape[1]}")
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
            adata = adata[:, adata.var.highly_variable]

        #TODO: Make this filtering apply only for datasets of genetic perturbation
        #We remove observations perturbed with a gene whuch is not in the columns of the dataset
        #We also remove perturbations that do not have any embedding
        if self.config.etl_config.drop_unmatched_perts:
            logger.info("Removing observations with perturbations not in the dataset as a column")

            number_perts_before = len(adata.obs[PERT_KEY].unique())
            adata = adata[((adata.obs[PERT_KEY] == CONTROL_PERT) | adata.obs[PERT_KEY].isin(adata.var_names))]
            number_perts_after_column_matching = len(adata.obs[PERT_KEY].unique())

            pert_embedding = adata.uns[PERT_EMBEDDING_KEY]
            perts_with_embedding = set(pert_embedding.keys())



            adata = adata[(adata.obs[PERT_KEY].isin(perts_with_embedding)) | (adata.obs[PERT_KEY] == CONTROL_PERT)]
            number_perts_after_embedding_matching = len(adata.obs[PERT_KEY].unique())

            logger.info(f"Removed {number_perts_before - number_perts_after_column_matching} perturbations that were not in the dataset columns and {number_perts_after_column_matching - number_perts_after_embedding_matching} perturbations that did not have an embedding for a total of {number_perts_before - number_perts_after_embedding_matching} perturbations removed out of an initial {number_perts_before} perturbations")
        
        
        adata.X = adata.X.toarray().astype('float32')
        # Set gene names
        if dataset_details.var_names_key:
            adata.var_names = adata.var[dataset_details.var_names_key]

        # Apply normalization and log1p if needed
        if self.config.etl_config.count_norm & (not dataset_details.count_normalized):
            sc.pp.normalize_total(adata, target_sum=10_000)
        elif ((not self.config.etl_config.count_norm) & dataset_details.count_normalized):
            raise ValueError("Specified dataset is count normalized, but normalization is turned off in the config")
        
        if self.config.etl_config.log1p & (not dataset_details.log1p_transformed):
            sc.pp.log1p(adata)
        elif (not self.config.etl_config.log1p) & dataset_details.log1p_transformed:
            raise ValueError("Specified dataset is log1p transformed, but log1p transformation is turned off in the config")

        if self.metric_space is not None:
            if self.metric_space in dataset_details.metric_spaces:
                adata.obsm["metric_space"] = adata.obsm[self.metric_space]
            else:
                raise ValueError(f"Metric space {self.config.embedding_config.metric_space} not found in metric spaces available for dataset {dataset_details.name}")


        #Precomputed DEGs eg for NN Oracle
        if dataset_details.precomputed_DEGs:
            DEGs = json.load(open(f"{dataset_details.folder_path}/DEGs.json"))
            adata.uns["DEGs"] = DEGs


        #Synthetic config is specified
        if self.config.etl_config.synthetic is not None:
            model_config_path = self.config.etl_config.synthetic.model_config_path
            synthetic_model_config = ModelConfig.from_yaml(Path(model_config_path).resolve())

            #Fetch the training config for the synthetic data, ETL and Split config should be the same, modulo the synthetic part
            synthetic_data_config = self.config.etl_config.copy()
            synthetic_data_config.synthetic = None

            synthetic_datasplit_config = self.config.datasplit_config.copy()


            #Config that should have been used to generate the synthetic data
            synthetic_data_config = Config(model_config=synthetic_model_config,
                                            etl_config=synthetic_data_config,
                                            datasplit_config=synthetic_datasplit_config)


            #We verify that this exact config exists for our dataset
            if synthetic_data_config.get_synthetic_config_ID() not in self.training_dataset_details.synthetic_versions:
                raise ValueError(f"Could not find a config with name {synthetic_data_config.get_synthetic_config_ID()} for dataset {self.training_dataset_details.name}, please check that the synthetic data was generated with the same config.")

            #We load the synthetic data
            synthetic_data_path = f"{dataset_details.folder_path}/synthetic_data/{synthetic_data_config.get_synthetic_config_ID()}"

            synthetic_data_files = os.listdir(synthetic_data_path)

            #We get the paths of all the files in this folder
            synthetic_data_paths = [Path(f"{synthetic_data_path}/{file}").resolve() for file in synthetic_data_files]

            #We add them to the adata
            adata.uns[SYNTHETIC_DATA_PATHS_KEY] = synthetic_data_paths

        return adata

    def get_training_data(self) -> Tuple[sc.AnnData, Optional[dict]]:
        """
        Returns the training data according to the config.
        If an pert embedding is specified then it is also returned
        """

        # Checking if we have already a cached version of the preprocessed training data
        if self.complete_training_adata is None:
            logger.info(f"Loading training data at path: {self.training_dataset_details.path}")
            with open(self.training_dataset_details.path, 'rb') as f:
                adata = sc.read_h5ad(f)

            logger.info(f"Loaded unpreprocessed data, # of data points: {len(adata)}, # of genes: {len(adata.var)}.")

            logger.info("Preprocessing training data")
            adata = self.preprocess_data(adata, training=True)

            self.complete_training_adata = adata
            logger.debug(f"Loaded complete data, # of data points: {len(adata)}, # of genes: {len(adata.var)}, # of conditions: {len(adata.obs[PERT_KEY].unique())}")

    
        # Doing the data split
        if self.config.datasplit_config.mode == "ood":
            logger.info("Doing OOD split")
            # Taking cells for training where neither the cell nor the perturbation is held out
            holdout_mask = (self.complete_training_adata.obs[PERT_KEY].isin(self.config.datasplit_config.holdout_perts)) | (self.complete_training_adata.obs[CELL_KEY].isin(self.config.datasplit_config.holdout_cells))
            train_mask = ~holdout_mask

        # IID, we include unperturbed holdouts cells
        else:
            logger.info("Doing IID split")
            # Holding out only cells that have heldout perturbations AND cell. Thus:
            # A perturbation will be included on the non holdout cell type eg
            # Control of heldout cell type will be included
            holdout_mask = (self.complete_training_adata.obs[CELL_KEY].isin(self.config.datasplit_config.holdout_cells)) & (self.complete_training_adata.obs[PERT_KEY].isin(self.config.datasplit_config.holdout_perts))
            train_mask = ~holdout_mask

        #Subsetting complete dataset to entries for training
        adata_train = self.complete_training_adata[train_mask]


        #We still return the pert embedding here to not break code that relies on it
        pert_embedding = adata.uns[PERT_EMBEDDING_KEY]
        return adata_train, pert_embedding

    def get_complete_training_dataset(self) -> sc.AnnData:
        """
        Returns the entire dataset for training, including the heldout cells and perturbations.
        Data is preprocessed according to config
        """
        if self.complete_training_adata is None:
            with open(self.training_dataset_details.path, 'rb') as f:
                adata= sc.read_h5ad(f)
            logger.info("Preprocessing training data")
            self.complete_training_adata = self.preprocess_data(adata, training=True)
 
        return self.complete_training_adata

    def get_eval_data(self):
        self.eval_dataset_details = Catalogue.get_dataset_details(self.config.eval_config.dataset)

        #To avoid loading the same data twice
        if self.config.datasplit_config.dataset == self.config.eval_config.dataset:
            self.complete_eval_adata = self.get_complete_training_dataset()

        else:

            logger.info(f"Loading evaluation data at path: {self.eval_dataset_details.path}")
            with open(self.eval_dataset_details.path, 'rb') as f:
                adata= sc.read_h5ad(f)

            logger.info("Preprocessing evaluation data")
            adata = self.preprocess_data(adata, training=False)
            self.complete_eval_adata = adata

        #TODO: Double check that this is evaluated lazily
        logger.debug(f"Eval targets are {self.config.eval_config.evaluation_targets}")
        for cell_id, pert_id in self.config.eval_config.evaluation_targets:
            gt_data = self.complete_eval_adata[(self.complete_eval_adata.obs[PERT_KEY] == pert_id) & (self.complete_eval_adata.obs[CELL_KEY] == cell_id)]
            ctrl_data = self.complete_eval_adata[(self.complete_eval_adata.obs[CELL_KEY] == cell_id) & (self.complete_eval_adata.obs[PERT_KEY] == CONTROL_PERT)]

            #If no data is found we skip the evaluation
            if len(gt_data) == 0:
                logger.warning(f"No data found for cell: {cell_id}, pert: {pert_id} in {self.config.eval_config.dataset}, will skip evaluation")
                continue
            
            if len(ctrl_data) == 0:
                logger.critical(f"No control data found for cell: {cell_id} in {self.config.eval_config.dataset}, will skip evaluation")
                continue
           
            
            yield cell_id, pert_id, ctrl_data, gt_data



        
