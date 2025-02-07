
from curses.ascii import CAN
import json
from dataclasses import dataclass, field, asdict
from turtle import st
from typing import List, Optional
from pyparsing import C
import yaml
import os
from pathlib import Path
from omnicell.constants import DATA_CATALOGUE_PATH

#How to we save, load and update the catalogue
#Who handles it, who has visibility over what
#How do we handle edition of the catalogue
#How do we type a dataset entry

import os
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatasetDetails:
    path: str
    folder_path: str
    cell_key: str
    control: str
    pert_key: str
    var_names_key: str
    HVG: bool
    log1p_transformed: bool
    count_normalized: bool
    precomputed_DEGs: bool
    description: Optional[str] = None
    pert_embeddings: List[str] = field(default_factory=list)
    cell_embeddings: List[str] = field(default_factory=list)
    gene_embeddings: List[str] = field(default_factory=list)


    def to_dict(self):
        return asdict(self)


"""Static class, handles all read and writes atomically"""
class Catalogue: 

    @staticmethod
    def _get_catalogue():
        _catalogue = {}
        for fp in os.listdir(DATA_CATALOGUE_PATH):
            file_path = os.path.join(DATA_CATALOGUE_PATH, fp)

            logger.info(f"Loading data catalogue from {file_path}")
            if file_path.endswith(".yaml"):
                with open(file_path) as f:
                    file_name = os.path.basename(fp)
                    dataset_name = file_name.split(".")[0]
                    _catalogue[dataset_name] = DatasetDetails(**yaml.load(f, Loader=yaml.FullLoader))

        return _catalogue



    @staticmethod
    def get_dataset_details(dataset_name) -> DatasetDetails:
        catalogue = Catalogue._get_catalogue()

        if dataset_name in catalogue:
            return catalogue[dataset_name]
        
        else:
            raise ValueError(f"Dataset {dataset_name} not found in catalogue")
        
    @staticmethod
    def get_dataset_names():
        return [x['name'] for x in Catalogue._get_catalogue()['datasets']]

    "Might be useful for some script down the line"
    def register_new_dataset(self, name, dd: DatasetDetails):
        raise NotImplementedError("Not implemented yet")



    @staticmethod
    def register_new_pert_embedding(dataset_name, embedding_name):
        catalogue = Catalogue._get_catalogue()
        if dataset_name in catalogue:
            if embedding_name not in catalogue[dataset_name].pert_embeddings:
                catalogue[dataset_name].pert_embeddings.append(embedding_name)
                Catalogue._save(catalogue)
            else: 
                raise ValueError(f"Pert Embedding {embedding_name} already exists for dataset {dataset_name}")
        else:
            raise ValueError(f"Dataset {dataset_name} not found in catalogue")

    @staticmethod
    def register_new_cell_embedding(dataset_name, embedding_name):
        catalogue = Catalogue._get_catalogue()
        if dataset_name in catalogue:
            if embedding_name not in catalogue[dataset_name].cell_embeddings:
                catalogue[dataset_name].cell_embeddings.append(embedding_name)
                Catalogue._save(catalogue)
            else: 
                raise ValueError(f"Embedding {embedding_name} already exists for dataset {dataset_name}")
        else:
            raise ValueError(f"Dataset {dataset_name} not found in catalogue")

    @staticmethod
    def register_new_gene_embedding(dataset_name, embedding_name):
        catalogue = Catalogue._get_catalogue()
        if dataset_name in catalogue:
            if embedding_name not in catalogue[dataset_name].gene_embeddings:
                catalogue[dataset_name].gene_embeddings.append(embedding_name)
                Catalogue._save(catalogue)
            else: 
                raise ValueError(f"Gene Embedding {embedding_name} already exists for dataset {dataset_name}")
        else:
            raise ValueError(f"Dataset {dataset_name} not found in catalogue")

    @staticmethod
    def set_DEGs_available(dataset_name):
        catalogue = Catalogue._get_catalogue()
        if dataset_name in catalogue:
            catalogue[dataset_name].precomputed_DEGs = True
            Catalogue._save(catalogue)
        else:
            raise ValueError(f"Dataset {dataset_name} not found in catalogue")

            
    """Flushes the content of the catalogue to disk, pushing any changes that have been made"""
    @staticmethod
    def _save(catalogue: dict):
        for dataset_name, dataset_details in catalogue.items():
            with open(f"{DATA_CATALOGUE_PATH}/{dataset_name}.yaml", "w") as f:
                yaml.dump(dataset_details.to_dict(), f, default_flow_style=False)

