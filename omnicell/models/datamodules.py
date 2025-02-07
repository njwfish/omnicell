import torch

from torch.utils.data import Sampler
from typing import Iterator, List

import numpy as np
import pandas as pd
from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY
from omnicell.models.collate_fns import ot_collate, cfm_collate
from collections import defaultdict


class StratifiedBatchSampler(Sampler[List[int]]):
    def __init__(
        self, ns, batch_size: int
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        
        self.num_strata = np.prod(ns.shape)
        self.ns = ns
        self.probs = ns.flatten() / np.sum(ns)
        print("Strata probs", np.sort(self.probs))
        self.batch_size = batch_size
        self.batch_sizes = np.minimum(ns, batch_size)
    
    def get_random_stratum(self):
        linear_idx = np.random.choice(self.num_strata, p=self.probs)
        stratum = np.unravel_index(linear_idx, self.ns.shape)
        return stratum

    # def __iter__(self) -> Iterator[List[int]]:
    #     # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
    #    while True:
    #         stratum = self.get_random_stratum()
    #         try:
    #             batch_stratum = np.repeat(np.array(stratum)[None, :], self.batch_sizes[stratum], axis=0)
    #             batch = np.random.choice(self.ns[stratum], self.batch_sizes[stratum], replace=False)
    #             yield zip(batch_stratum, batch)
    #         except StopIteration:
    #             break


    def __iter__(self) -> Iterator[List[int]]:
        # Calculate number of batches based on total samples and batch size
        samples_remaining = np.sum(self.ns)
        while True:
            if samples_remaining < 0:
                break
            stratum = self.get_random_stratum()
            batch_stratum = np.repeat(np.array(stratum)[None, :], self.batch_sizes[stratum], axis=0)
            batch = np.random.choice(self.ns[stratum], self.batch_sizes[stratum], replace=False)
            samples_remaining -= batch.shape[0]
            yield zip(batch_stratum, batch)

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return np.sum(self.ns) // self.batch_size

class SCFMDataset(torch.utils.data.Dataset):
    def __init__(
        self, source, target, pert_ids, pert_map, source_strata, target_strata
    ):
        source, target = np.array(source), np.array(target)
        pert_ids = np.array(pert_ids) # , np.array(pert_mat)
        
        assert target.shape[0] == pert_ids.shape[0]
        assert source.shape[0] == source_strata.shape[0]
        assert target.shape[0] == target_strata.shape[0]
        
        self.size = target.shape[0]
        self.source_strata = source_strata
        self.target_strata = target_strata
        self.strata = np.unique(source_strata)
        self.num_strata = len(self.strata)
        print(source_strata, self.strata, self.num_strata)
        
        self.unique_pert_ids = np.unique(pert_ids)
        
        print("Creating source indices")
        self.source_indices = {
            stratum: np.where(source_strata == stratum)[0] 
            for stratum in self.strata
        }
        print("Creating target indices")
        self.target_indices = {
            stratum: np.where(target_strata == stratum)[0] 
            for stratum in self.strata
        }
        print("Creating pert indices")
        self.pert_indices = {
            pert: np.where(pert_ids == pert)[0] 
            for pert in self.unique_pert_ids
        }
        print("Creating source and target dicts")
        self.source = {}
        self.target = defaultdict(dict)
        self.pert_ids = defaultdict(dict)
        for stratum in self.strata:
            self.source[stratum] = source[source_strata == stratum]
            for pert in self.unique_pert_ids:
                self.target[stratum][pert] = target[(target_strata == stratum) & (pert_ids == pert)]
                self.pert_ids[stratum][pert] = pert_ids[(target_strata == stratum) & (pert_ids == pert)]
        self.pert_map = pert_map
        

    def __len__(self):
        return self.size

    def __getitem__(self, strata_idx):
        (stratum_idx, pert_idx), idx = strata_idx
        stratum, pert = self.strata[stratum_idx], self.unique_pert_ids[pert_idx]
        sidx = np.random.choice(self.source[stratum].shape[0])
        print()
        return (
            self.source[stratum][sidx],
            self.target[stratum][pert][idx],
            self.pert_map[self.pert_ids[stratum][pert][idx]],
        )

def get_identity_features(adata, cell_type_features=True):
    perts = pd.get_dummies(adata.obs[PERT_KEY]).values.astype(float)
    cell_types = pd.get_dummies(adata.obs[CELL_KEY]).values
    if cell_type_features:
        combo = pd.get_dummies(adata.obs[CELL_KEY].astype(str) + adata.obs[PERT_KEY].astype(str)).values
        idx = (combo!=0).argmax(axis=0)
        pert_mat = np.hstack([cell_types, perts])[idx, :].astype('float32')
    else:
        combo = perts
        idx = (combo!=0).argmax(axis=0)
        pert_mat = perts[idx, :].astype('float32')
    
    pert_ids = combo.argmax(axis=1)
    cell_types = cell_types.argmax(axis=1)
    return pert_ids, pert_mat, cell_types


def get_dataloader(
        adata, batch_size=512, verbose=0, pert_map=None, pert_ids=None, collate='ot', X=None
):
        if X is None:
            X = adata.X.toarray()
            
        # CONTROL_PERT = 'NT'
        control_idx = adata.obs[PERT_KEY] == CONTROL_PERT
        pert_idx = adata.obs[PERT_KEY] != CONTROL_PERT
        cell_types = adata.obs[CELL_KEY].values

        if pert_map is None:
            pert_ids, pert_map, cell_types = get_identity_features(adata)

        control_train = X[control_idx]
        pert_train = X[pert_idx]
        pert_ids_train =  pert_ids[pert_idx]
        control_cell_types = cell_types[control_idx]
        pert_cell_types = cell_types[pert_idx]

        # print("pert_ids", pert_ids, pert_ids.shape)
        # print("pert_ids_train", pert_ids_train, pert_ids_train.shape)

        # print("cell_types", cell_types, cell_types.shape)
        # print("control_cell_types", control_cell_types, control_cell_types.shape)
        # print("pert_cell_types", pert_cell_types, pert_cell_types.shape)        

        if collate == 'ot':
             collate_fn = ot_collate
        elif collate == 'cfm':
            collate_fn = cfm_collate    
        elif collate == 'cfm_control':
            collate_fn = lambda batch: cfm_collate(batch, return_control=True)
        else:
            raise ValueError(f"Collate function {collate} not recognized")

        dset = SCFMDataset(
            control_train, pert_train, 
            pert_ids_train, pert_map, 
            control_cell_types, pert_cell_types
        )
        ns = np.array(
            [
                [
                    dset.target[stratum][pert].shape[0] for pert in dset.target[stratum]
                ] for stratum in dset.target
            ]
        )
        
        dl = torch.utils.data.DataLoader(
            dset, collate_fn=collate_fn, 
            batch_sampler=StratifiedBatchSampler(
                ns=ns, batch_size=batch_size
            )
        )
        return dset, ns, dl
