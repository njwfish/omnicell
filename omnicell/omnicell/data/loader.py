import scanpy as sc
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from omnicell.config.config import Config
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
from omnicell.data.catalogue import DatasetDetails, Catalogue
import torch
import logging
import numpy as np
import json
import pandas as pd

import os


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
    perts = np.unique(adata.obs[PERT_KEY])
    # one hot encode set perts
    pert_rep = pd.get_dummies(perts).set_index(perts).astype(np.float32)
    return {pert: pert_rep[pert].values for pert in perts}


class DataLoader:
    def __init__(self, config: Config, data_catalogue: Catalogue):
        self.config = config
        self.data_catalogue = data_catalogue
        self.training_dataset_details: DatasetDetails = data_catalogue.get_dataset_details(config.get_training_dataset_name())

        logger.debug(f"Training dataset details: {self.training_dataset_details}")

        self.pert_embedding_name: Optional[str] = config.get_pert_embedding_name()

        self.cell_embedding_name: Optional[str] = config.get_cell_embedding_name()

        self.gene_embedding_name: Optional[str] = config.get_gene_embedding_name()
        
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
        # Standardize column names and key values
        condition_key = dataset_details.pert_key
        cell_key = dataset_details.cell_key if training else self.eval_dataset_details.cell_key
        control = dataset_details.control if training else self.eval_dataset_details.control

        #TODO: If we could rename the columns it would be better

        adata.obs.rename(columns={condition_key: PERT_KEY, cell_key: CELL_KEY}, inplace=True)

        adata.obs[PERT_KEY] = adata.obs[PERT_KEY].cat.rename_categories({control: CONTROL_PERT})

        if (self.config.get_cell_embedding_name() is not None) & (self.config.get_apply_normalization() | self.config.get_apply_log1p()):
            raise ValueError("Cannot both apply cell embedding and normalization/log1p transformation")
        
        elif self.config.get_cell_embedding_name() is not None:
            if self.config.has_local_cell_embedding:
                logger.info(f"Loading cell embedding from {self.config.get_cell_embedding_name()}")

                #TODO: This is something I will need to change
                adata.obsm["embedding"] = np.load(self.config.get_local_cell_embedding_path())
                
            elif self.config.get_cell_embedding_name() in dataset_details.cell_embeddings:            
                #We replace the data matrix with the cell embeddings
                adata.obsm["embedding"] = adata.obsm[self.config.get_cell_embedding_name()]
            else:
                raise ValueError(f"Cell embedding {self.config.get_cell_embedding_name()} not found in embeddings available for dataset {dataset_details.name}")
        else:
            adata.obsm["embedding"] = adata.X.toarray().astype('float32')
            # Set gene names
            if dataset_details.var_names_key:
                adata.var_names = adata.var[dataset_details.var_names_key]

            # Apply normalization and log1p if needed
            if self.config.get_apply_normalization() & (not dataset_details.count_normalized):
                sc.pp.normalize_total(adata, target_sum=10_000)
            elif not self.config.get_apply_normalization() & dataset_details.count_normalized:
                raise ValueError("Specified dataset is count normalized, but normalization is turned off in the config")
            
            if self.config.get_apply_log1p() & (not dataset_details.log1p_transformed):
                sc.pp.log1p(adata)
            elif not self.config.get_apply_log1p() & dataset_details.log1p_transformed:
                raise ValueError("Specified dataset is log1p transformed, but log1p transformation is turned off in the config")


        if self.config.get_metric_space() is not None:
            if self.config.get_metric_space() in dataset_details.metric_spaces:
                adata.obsm["metric_space"] = adata.obsm[self.config.get_metric_space()]
            else:
                raise ValueError(f"Metric space {self.config.get_metric_space()} not found in metric spaces available for dataset {dataset_details.name}")

        if self.gene_embedding_name is not None:
            if self.gene_embedding_name not in self.dataset_details.gene_embedding:
                raise ValueError(f"Gene Embedding {self.gene_embedding_name} is not found in gene embeddings available for dataset {dataset_details.name}")
            else:
                embedding = torch.load(f"{dataset_details.folder_path}/{self.gene_embedding_name}")
                adata.varm["gene_embedding"] = embedding.numpy()


        if dataset_details.precomputed_DEGs:
            DEGs = json.load(open(f"{dataset_details.folder_path}/DEGs.json"))
            adata.uns["DEGs"] = DEGs

        return adata

    def get_training_data(self) -> Tuple[sc.AnnData, Optional[dict]]:
        """
        Returns the training data according to the config.
        If an pert embedding is specified then it is also returned
        """

        # Checking if we have already a cached version of the training data
        if self.complete_training_adata is None:
            logger.info(f"Loading training data at path: {self.training_dataset_details.path}")
            # adata = sc.read(self.training_dataset_details.path)
            with open(self.training_dataset_details.path, 'rb') as f:
                adata = sc.read_h5ad(f)

            logger.info("Preprocessing training data")
            adata = self.preprocess_data(adata, training=True)

            self.complete_training_adata = adata
            logger.debug(f"Loaded complete data, # of data points: {len(adata)}, # of genes: {len(adata.var)}, # of conditions: {len(adata.obs[PERT_KEY].unique())}")

        #Getting the per embedding if it is specified
        if self.pert_embedding_name is not None:
            if self.pert_embedding_name not in self.training_dataset_details.pert_embeddings:
                raise ValueError(f"Perturbation embedding {self.pert_embedding_name} not found in embeddings available for dataset {self.training_dataset_details.name}")
            else:
                logger.info(f"Loading perturbation embedding from {self.training_dataset_details.folder_path}/{self.pert_embedding_name}.pt")
                pert_embedding = torch.load(f"{self.training_dataset_details.folder_path}/{self.pert_embedding_name}.pt")
        else:
            pert_embedding = get_identity_features(self.complete_training_adata)

        # Doing the data split
        if self.config.get_mode() == "ood":
            logger.info("Doing OOD split")
            # Taking cells for training where neither the cell nor the perturbation is held out
            holdout_mask = (self.complete_training_adata.obs[PERT_KEY].isin(self.config.get_heldout_perts())) | (self.complete_training_adata.obs[CELL_KEY].isin(self.config.get_heldout_cells()))
            train_mask = ~holdout_mask

        # IID, we include unperturbed holdouts cells
        else:
            logger.info("Doing IID split")
            # Holding out only cells that have heldout perturbations AND cell. Thus:
            # A perturbation will be included on the non holdout cell type eg
            # Control of heldout cell type will be included
            holdout_mask = (self.complete_training_adata.obs[CELL_KEY].isin(self.config.get_heldout_cells())) & (self.complete_training_adata.obs[PERT_KEY].isin(self.config.get_heldout_perts()))
            train_mask = ~holdout_mask

        #Subsetting complete dataset to entries for training
        adata_train = self.complete_training_adata[train_mask]

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
        self.eval_dataset_details = self.data_catalogue.get_dataset_details(self.config.get_eval_dataset_name())

        #To avoid loading the same data twice
        if self.config.get_training_dataset_name() == self.config.get_eval_dataset_name():
            self.complete_eval_adata = self.get_complete_training_dataset()

        else:

            logger.info(f"Loading evaluation data at path: {self.eval_dataset_details.path}")
            with open(self.eval_dataset_details.path, 'rb') as f:
                adata= sc.read_h5ad(f)

            logger.info("Preprocessing evaluation data")
            adata = self.preprocess_data(adata, training=False)
            self.complete_eval_adata = adata



        #TODO: Double check that this is evaluated lazily
        logger.debug(f"Eval targets are {self.config.get_eval_targets()}")
        for cell_id, pert_id in self.config.get_eval_targets():
            gt_data = self.complete_eval_adata[(self.complete_eval_adata.obs[PERT_KEY] == pert_id) & (self.complete_eval_adata.obs[CELL_KEY] == cell_id)]
            ctrl_data = self.complete_eval_adata[(self.complete_eval_adata.obs[CELL_KEY] == cell_id) & (self.complete_eval_adata.obs[PERT_KEY] == CONTROL_PERT)]



            #If no data is found we skip the evaluation
            if len(gt_data) == 0:
                logger.warning(f"No data found for cell: {cell_id}, pert: {pert_id} in {self.config.get_eval_dataset_name()}, will skip evaluation")
                continue
            
            if len(ctrl_data) == 0:
                logger.critical(f"No control data found for cell: {cell_id} in {self.config.get_eval_dataset_name()}, will skip evaluation")
                continue
           
            
            yield cell_id, pert_id, ctrl_data, gt_data



        
