
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml
import os
from pathlib import Path

#How to we save, load and update the catalogue
#Who handles it, who has visibility over what
#How do we handle edition of the catalogue
#How do we type a dataset entry

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

class Catalogue: 
    def __init__(self, path_to_catalogue):
        self.path = path_to_catalogue
        self._catalogue = {}



        #Iterate over all files in the catalogue directory
        for fp in os.listdir(self.path):
            file_path = os.path.join(self.path, fp)
            if file_path.endswith(".yaml"):
                with open(file_path) as f:
                    file_name = os.path.basename(fp)
                    dataset_name = file_name.split(".")[0]
                    self._catalogue[dataset_name] = DatasetDetails(**yaml.load(f, Loader=yaml.FullLoader))




    def get_dataset_details(self, dataset_name) -> DatasetDetails:
        if dataset_name in self._catalogue:
            return self._catalogue[dataset_name]
        
        else:
            raise ValueError(f"Dataset {dataset_name} not found in catalogue")
        

    def get_dataset_names(self):
        return [x['name'] for x in self.catalogue['datasets']]

    "Might be useful for some script down the line"
    def register_new_dataset(self, name, dd: DatasetDetails):
        raise NotImplementedError("Not implemented yet")



    #TODO: Misleading error, if we modify disk and call this after, disk is modified but still throws error
    def register_new_pert_embedding(self, dataset_name, embedding_name):
        if dataset_name in self._catalogue:
            if embedding_name not in self._catalogue[dataset_name].pert_embeddings:

                self._catalogue[dataset_name].pert_embeddings.append(embedding_name)
                self._save()
            else: 
                raise ValueError(f"Pert Embedding {embedding_name} already exists for dataset {dataset_name}")
        else:
            raise ValueError(f"Dataset {dataset_name} not found in catalogue")
        

    """Registers a new embedding for a dataset, will modify the corresponding catalogue entry, does save the data"""
    def register_new_cell_embedding(self, dataset_name, embedding_name):
        if dataset_name in self._catalogue:
            if embedding_name not in self._catalogue[dataset_name].cell_embeddings:

                self._catalogue[dataset_name].cell_embeddings.append(embedding_name)
                self._save()
            else: 
                raise ValueError(f"Embedding {embedding_name} already exists for dataset {dataset_name}")
        else:
            raise ValueError(f"Dataset {dataset_name} not found in catalogue")
        
    
    def register_new_gene_embedding(self, dataset_name, embedding_name):
        if dataset_name in self._catalogue:
            if embedding_name not in self._catalogue[dataset_name].gene_embeddings:

                self._catalogue[dataset_name].gene_embeddings.append(embedding_name)
                self._save()
            else: 
                raise ValueError(f"Gene Embedding {embedding_name} already exists for dataset {dataset_name}")
        else:
            raise ValueError(f"Dataset {dataset_name} not found in catalogue")


    """Flushes the content of the catalogue to disk, pushing any changes that have been made"""
    def _save(self):
        for dataset_name, dataset_details in self._catalogue.items():
            with open(f"{self.path}/{dataset_name}.yaml", "w") as f:
                yaml.dump(dataset_details.to_dict(), f, default_flow_style=False)

