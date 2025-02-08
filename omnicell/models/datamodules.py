import torch
import os

from torch.utils.data import Sampler
from typing import Iterator, List

import numpy as np
import pandas as pd
from omnicell.constants import CELL_KEY, CONTROL_PERT, PERT_KEY
from omnicell.models.collate_fns import ot_collate, cfm_collate, collate
from collections import defaultdict

from pathlib import Path


class StratifiedBatchSampler(Sampler[List[int]]):
    def __init__(
        self, ns, batch_size: int, samples_per_epoch: int = None
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        
        self.num_strata = np.prod(ns.shape)
        self.ns = ns
        self.probs = ns.flatten() / np.sum(ns)
        print("Strata probs", np.sort(self.probs))
        self.batch_size = batch_size
        self.batch_sizes = np.minimum(ns, batch_size)
        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
        else:
            self.samples_per_epoch = np.sum(self.ns)
    
    def get_random_stratum(self):
        linear_idx = np.random.choice(self.num_strata, p=self.probs)
        stratum = np.unravel_index(linear_idx, self.ns.shape)
        return stratum

    def __iter__(self) -> Iterator[List[int]]:
        # Calculate number of batches based on total samples and batch size
        samples_remaining = self.samples_per_epoch
        while True:
            if samples_remaining < 0:
                break
            stratum = self.get_random_stratum()
            batch_stratum = np.repeat(np.array(stratum)[None, :], self.batch_sizes[stratum], axis=0)
            batch = np.random.choice(self.ns[stratum], self.batch_sizes[stratum], replace=False)
            samples_remaining -= batch.shape[0]
            yield zip(batch_stratum, batch)

    def __len__(self) -> int:
        return np.sum(np.minimum(self.ns // self.batch_size, 1))

class OnlinePairedStratifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self, source, target, pert_ids, pert_map, source_strata, target_strata
    ):
        source, target = np.array(source).astype(np.float32), np.array(target).astype(np.float32)
        pert_ids = np.array(pert_ids)
        
        assert target.shape[0] == pert_ids.shape[0]
        assert source.shape[0] == source_strata.shape[0]
        assert target.shape[0] == target_strata.shape[0]
        
        self.size = target.shape[0]
        self.source_strata = source_strata
        self.target_strata = target_strata
        self.strata = np.unique(source_strata)
        self.num_strata = len(self.strata)
        
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

        # Compute ns_2d for stratification
        self.ns = np.array([
            [
                self.target[stratum][pert].shape[0]
                for pert in self.unique_pert_ids
            ] for stratum in self.strata
        ])
        
        self.samples_per_epoch = np.sum(self.ns)
        

    def __len__(self):
        return self.size

    def __getitem__(self, strata_idx):
        (stratum_idx, pert_idx), idx = strata_idx
        stratum, pert = self.strata[stratum_idx], self.unique_pert_ids[pert_idx]
        sidx = np.random.choice(self.source[stratum].shape[0])
        return (
            self.source[stratum][sidx],
            self.target[stratum][pert][idx],
            self.pert_map[self.pert_ids[stratum][pert][idx]],
        )

class StreamingOnlinePairedStratifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir, pert_map, num_files, device=None
    ):
        """Dataset that maintains online pairing structure but streams from files.
        
        Args:
            data_dir: Directory containing the pkl files
            pert_map: Dictionary mapping perturbation IDs to embeddings
            num_files: Total number of files to cycle through
            device: torch device (defaults to cuda if available)
        """
        self.data_dir = Path(data_dir)
        self.pert_map = pert_map
        self.num_files = num_files
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load first file to get structure
        first_file = self._load_file(0)
        self.source_structure = first_file['source']
        self.target_structure = first_file['synthetic_counterfactuals']
        
        # Setup basic structure info
        self.strata = np.array(list(self.source_structure.keys()))
        self.unique_pert_ids = np.array(list(self.target_structure[self.strata[0]].keys()))
        
        # Compute ns_2d for stratification
        # TODO: this may cause problems if the perts per file or number of perts per file differ
        raise ValueError("Not implemented, see above TODO")
        self.ns = np.array([
            [
                self.target_structure[stratum][pert].shape[0]
                for pert in self.unique_pert_ids
            ] for stratum in self.strata
        ])
        
        self.samples_per_epoch = np.sum(self.ns)
        
        # Cache for loaded files
        self._cache = {}
        self._cache_size = 3
        self.current_file_idx = -1  # Track current file
        self._preload_next_file()  # Preload first file
        self.current_file_idx = 0
        
    def _load_file(self, file_idx):
        """Load a specific file by index"""
        file_path = self.data_dir / f"synthetic_counterfactuals_{512 * file_idx}.pkl"
        return np.load(file_path, allow_pickle=True)
    
    def _preload_next_file(self):
        """Preload the next file in sequence"""
        next_idx = (self.current_file_idx + 1) % self.num_files
        if next_idx not in self._cache and len(self._cache) >= self._cache_size:
            # Remove oldest file if not current file
            keys_to_remove = [
                k for k in self._cache.keys() 
                if k != self.current_file_idx
            ][:1]
            if keys_to_remove:
                del self._cache[keys_to_remove[0]]
        
        if next_idx not in self._cache:
            self._cache[next_idx] = self._load_file(next_idx)
    
    def _advance_file(self):
        """Move to next file and preload the next one"""
        self.current_file_idx = (self.current_file_idx + 1) % self.num_files
        self._preload_next_file()

    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, strata_idx):
        (stratum_idx, pert_idx), idx = strata_idx
        
        # Use current file
        data = self._cache[self.current_file_idx]
        
        stratum = self.strata[stratum_idx]
        pert = self.unique_pert_ids[pert_idx]
        
        # Random source sample from same stratum
        sidx = np.random.choice(len(data['source'][stratum]))
        
        return (
            data['source'][stratum][sidx].astype(np.float32),
            data['synthetic_counterfactuals'][stratum][pert][idx].astype(np.float32),
            self.pert_map[pert]
        )
    
    def on_epoch_end(self):
        """Call this at the end of each epoch to reset file counter"""
        self.current_file_idx = 0
        self._preload_next_file()

class StreamingOfflinePairedStratifiedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir,  # Directory containing the pkl files
            pert_map,
            num_files,  # Total number of files
            device=None
    ):
        """This dataset loads a set of files paired offline while preserving the stratification structure.
        For single cell data every batch will contains cells from the same cell type, but different perturbations.
        This is different from the OnlinePairedStratifiedDataset which requires batches to be from the same cell 
        type and same perturbation for the pairing.

        
        """
        self.data_dir = Path(data_dir)
        self.pert_map = pert_map
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_files = num_files
        
        # Load first file to get structure
        first_file = self._load_file(0)
        self.source_structure = first_file['source']
        self.target_structure = first_file['synthetic_counterfactuals']
        
        # Setup basic structure info
        self.strata = np.array(list(self.source_structure.keys()))
        self.unique_pert_ids = np.array(list(self.target_structure[self.strata[0]].keys()))
        
        self.ns = np.array([
            len(self.source_structure[stratum]) for stratum in self.strata
        ])
        
        self.samples_per_epoch = len(self.unique_pert_ids) * self.source_structure[self.strata[0]].shape[0] * self.num_files
        
        # Cache for loaded files
        self._cache = {}
        self._cache_size = 3
        self.current_file_idx = -1  # Track current file
        self._preload_next_file()  # Preload first file
        self.current_file_idx = 0
        
    def _load_file(self, file_idx):
        """Load a specific file by index"""
        file_path = self.data_dir / f"synthetic_counterfactuals_{512 * file_idx}.pkl"
        return np.load(file_path, allow_pickle=True)
    
    def _preload_next_file(self):
        """Preload the next file in sequence"""
        next_idx = (self.current_file_idx + 1) % self.num_files
        if next_idx not in self._cache and len(self._cache) >= self._cache_size:
            # Remove oldest file if not current file
            keys_to_remove = [
                k for k in self._cache.keys() 
                if k != self.current_file_idx
            ][:1]
            if keys_to_remove:
                del self._cache[keys_to_remove[0]]
        
        if next_idx not in self._cache:
            self._cache[next_idx] = self._load_file(next_idx)
    
    def _advance_file(self):
        """Move to next file and preload the next one"""
        self.current_file_idx = (self.current_file_idx + 1) % self.num_files

        self._preload_next_file()

    def __len__(self):
        return self.samples_per_epoch 
    
    def __getitem__(self, strata_idx):
        (stratum_idx,), idx = strata_idx
        
        # Use current file instead of random
        data = self._cache[self.current_file_idx]
        
        stratum = self.strata[stratum_idx]
        pert = np.random.choice(self.unique_pert_ids)
        
        # Return numpy arrays, let DataLoader handle device transfer
        return (
            data['source'][stratum][idx].astype(np.float32),
            data['synthetic_counterfactuals'][stratum][pert][idx].astype(np.float32),
            self.pert_map[pert]
        )
    
    def on_epoch_end(self):
        """Call this at the end of each epoch to reset file counter"""
        self.current_file_idx = 0
        self._preload_next_file()


def get_dataloader(
        adata, pert_map, pert_ids, offline=False, file_stream=None,
        batch_size=512, verbose=0, collate=None, X=None
):
        if collate is None:
            collate_fn = collate
        elif collate == 'ot':
             collate_fn = ot_collate
        elif collate == 'cfm':
            collate_fn = cfm_collate    
        elif collate == 'cfm_control':
            collate_fn = lambda batch: cfm_collate(batch, return_control=True)
        else:
            raise ValueError(f"Collate function {collate} not recognized")

        if file_stream:
            num_files = len(os.listdir(file_stream))
            if collate is not None:
                # TODO implement cfm collate for file streaming, should be easy to do
                raise ValueError("OT collate not supported for file streaming right now")
            if offline:
                dset = StreamingOfflinePairedStratifiedDataset(
                    data_dir=file_stream,
                    pert_map=pert_map,
                    num_files=num_files
                )
            else:
                dset = StreamingOnlinePairedStratifiedDataset(
                    data_dir=file_stream,
                    pert_map=pert_map,
                    num_files=num_files
                )
        else:
            if X is None:
                X = adata.X.toarray()
            
            control_idx = adata.obs[PERT_KEY] == CONTROL_PERT
            pert_idx = adata.obs[PERT_KEY] != CONTROL_PERT
            cell_types = adata.obs[CELL_KEY].values


            control_train = X[control_idx]
            pert_train = X[pert_idx]
            pert_ids_train =  pert_ids[pert_idx]
            control_cell_types = cell_types[control_idx]
            pert_cell_types = cell_types[pert_idx]
            if offline:
                raise ValueError("Offline pairing requires file streaming")
            else:
                dset = OnlinePairedStratifiedDataset(
                    control_train, pert_train, 
                    pert_ids_train, pert_map, 
                    control_cell_types, pert_cell_types
                )
        
        dl = torch.utils.data.DataLoader(
            dset, collate_fn=collate_fn, 
            batch_sampler=StratifiedBatchSampler(
                ns=dset.ns, batch_size=batch_size, samples_per_epoch=dset.samples_per_epoch
            )
        )
        return dset, dl
