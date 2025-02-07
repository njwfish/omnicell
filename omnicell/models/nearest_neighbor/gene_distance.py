import numpy
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from omnicell.constants import PERT_KEY, CELL_KEY, CONTROL_PERT
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class NearestNeighborPredictor():
    def __init__(self, config):
        self.config = config
        self.train_adata = None
        self.seen_cell_types = None
        self.seen_perts = None
        self.mean_shift = config['mean_shift']
        #TODO: We can compute means here for some extra perf

    def train(self, adata, model_savepath: Path):
        self.train_adata = adata
        self.seen_cell_types = adata.obs[CELL_KEY].unique()
        self.seen_perts = adata.obs[PERT_KEY].unique()

    
    def make_predict(self, adata: sc.AnnData, pert_id: str, cell_type: str) -> np.ndarray:
        assert self.train_adata is not None, "Model has not been trained yet"
        if cell_type in self.seen_cell_types:
            if pert_id in self.seen_perts:
                raise NotImplementedError("Both cell type and perturbation are in the training data, in distribution prediction not implemented yet")
            else:
                return self._predict_across_pert(adata, pert_id, cell_type)
        else:
            if pert_id in self.seen_perts:
                raise NotImplementedError("Gene distance version of NN can only do across perts, not across cells.")
            else:
                raise NotImplementedError("Both cell type and perturbation are not in the training data, out of distribution prediction not implemented yet")
               


    #SO I want to predict across genes --> Two options either we provide the data or we don't provide the data on which the prediction is made
    def _predict_across_pert(self, adata: sc.AnnData, target: str, cell_id: str) -> np.ndarray:
        """
        Makes a prediction for an unseen perturbation.
        
        Takes the perturbation effect in the training data which is closest to the heldout perturbation and applies it to the given control data

        Parameters
        ----------
        target : str
            The target perturbation to predict

        cell_id : str
            The cell type of the data on which the prediction is done

        adata : AnnData
            The AnnData object control data on the cell type of the prediction.

        Returns
        -------
        np.ndarray
            The predicted perturbation for the target using the control perturbation datapoints of the training.

        """

        assert self.train_adata is not None, "Model has not been trained yet"
        assert target not in self.train_adata.obs[PERT_KEY].unique(), "Target perturbation is already in the training data"
        assert adata.obs[CELL_KEY].nunique() == 1, "Heldout cell data must contain only one cell type"
        assert adata.obs[CELL_KEY].unique()[0] == cell_id, "Heldout cell data must contain only one cell type"
        assert adata.obs[PERT_KEY].nunique() == 1, "Heldout cell data must contain only control data"
        assert adata.obs[PERT_KEY].unique()[0] == CONTROL_PERT, "Heldout cell data must contain only control data"
        
        logger.debug(f'Predicting unseen perturbation {target} using all training data')
        
        num_of_degs = self.config['num_of_degs']

        unique_perturbations = self.train_adata.obs[PERT_KEY].unique()
        unique_genes_noholdout = [ug for ug in unique_perturbations if (ug!=target and ug!=CONTROL_PERT)]

        logger.debug(f"Len of unique genes_no_ho: {len(unique_genes_noholdout)}")

        DEGSlist = []
        GTOlist = []

        inp = self.train_adata[self.train_adata.obs[CELL_KEY] == cell_id].copy()
        
        
        #TODO: FIX, WILL ONLY WORK ON Seurat_IFNB.h5ad

        logger.debug(f'Input.var shape {inp.var.shape}')
        logger.debug(f"Input.var {inp.var}")        
        
        
        logger.debug(f' # Of cell with type {cell_id} in training data {len(inp)}')
        logger.debug(f'Finding nearest neighbor perturbation for {target}')
        logger.debug(f"inp.var_names shape {inp.var_names.shape}")

        logger.debug(f"Inp.var_name == target {inp.var_names==target}")

        logger.debug(f" Inp.var_name == target count {np.count_nonzero(inp.var_names==target)}")
        
        logger.info(f"Number of genes to compare to {len(unique_genes_noholdout)}")
        
        invalid_perts = []
        for ug in unique_genes_noholdout:
            try: 
                cont = np.array(inp[inp.obs[PERT_KEY] == CONTROL_PERT].obsm['embedding'])
                pert = np.array(inp[inp.obs[PERT_KEY] == ug].obsm['embedding'])
                
                logger.debug(f'Finding nearest neighbor perturbation for {target} using {ug}')
                logger.debug(f'Control shape {cont.shape}, pert shape {pert.shape}')
                
                control = sc.AnnData(X=cont)
                            
                true_pert = sc.AnnData(X=pert)
            
                
                control.obs_names = control.obs_names+'-1'

                control.obsm['embedding'][0,(control.obsm['embedding'].var(axis=0)==0)] += np.amin(control.obsm['embedding'][np.nonzero(control.obsm['embedding'])])

                #Bug Fixing: When a pert is not present on a cell type we ignore it.
                true_pert.obsm['embedding'][0,(true_pert.obsm['embedding'].var(axis=0)==0)] += np.amin(true_pert.obsm['embedding'][np.nonzero(true_pert.obsm['embedding'])])
                logger.warning(f"Error when computing DEG and GTO for {ug}")
                invalid_perts.append(ug)
                    


                
                temp_concat = sc.concat([control, true_pert], label = 'batch')
                sc.tl.rank_genes_groups(temp_concat, 'batch', method='wilcoxon', groups = ['1'], ref = '0')
                
                rankings = temp_concat.uns['rank_genes_groups']
                result_df = pd.DataFrame({'pva': rankings['pvals_adj']['1'], 'pvals_adj': rankings['scores']['1']}, index = rankings['names']['1'])
                result_df.index = result_df.index.astype(np.int32)
                result_df = result_df.sort_index()
                
                GTO = torch.from_numpy(np.array(result_df['pvals_adj']).astype(np.float32))
                
                DEGs = torch.argsort(torch.abs(GTO))
                
                logger.debug(f'GTO shape {GTO.shape}, DEGs shape {DEGs.shape}')
                DEGSlist.append(DEGs)
                GTOlist.append(GTO)

            except Exception as e:
                logger.warning(f"Error when computing DEG and GTO for {ug}")
                logger.warning(e)
                invalid_perts.append(ug)
                    

        logger.debug(f"DEGSlist shape {len(DEGSlist)}")
        logger.debug(f"GTOlist shape {len(GTOlist)}")
        significant_reducers = []
        significant_reducers_pval = []
        for genno in unique_genes_noholdout:

            #Bug Fixing: When a pert is not present on a cell type we ignore it. 
            if genno in invalid_perts:
                continue
            logger.debug(f'Finding nearest neighbor perturbation for {target} using {genno}')
            ug_index = unique_genes_noholdout.index(genno)
            logger.debug(f'ug_index {ug_index}')
            gto_gene = GTOlist[ug_index]
            mask1 = np.where(inp.var_names==target)[0][0]
            mask2 = torch.where(DEGSlist[unique_genes_noholdout.index(genno)] == mask1)[0][0].item()
            rank = gto_gene[mask2]
            if ((0 <= rank) and (rank <= self.config['pvalcut'])):
                significant_reducers.append(genno)
                significant_reducers_pval.append(rank) 

        reduced_DEGs = []    
        for sr in significant_reducers:
            reduced_DEGs.append(DEGSlist[unique_genes_noholdout.index(sr)][-num_of_degs:])
        reduced_DEGs = torch.cat(reduced_DEGs)   

        # Count the occurrences of each element
        unique_elements, counts = torch.unique(reduced_DEGs, return_counts=True)

        # Filter to keep only elements that repeat more than once
        repeated_elements = unique_elements[counts > 1]

        # Mask the original tensor to keep only repeated elements
        duplicated_DEGs = torch.unique(reduced_DEGs[torch.isin(reduced_DEGs, repeated_elements)])

        bestnnscore = 0
        bestpval = 0

        ranking = []

        
        for jq, sr in enumerate(significant_reducers):
            nnscore = len(np.intersect1d(duplicated_DEGs.cpu().detach().numpy(),DEGSlist[unique_genes_noholdout.index(sr)][-num_of_degs:].cpu().detach().numpy()))
            ranking.append([sr, nnscore])
            if nnscore > bestnnscore:
                nnbr = sr
                bestnnscore = nnscore
                bestpval = significant_reducers_pval[jq]
            #pval as tie-break
            if nnscore == bestnnscore:
                if bestpval > significant_reducers_pval[jq]:
                    nnbr = sr
                    bestnnscore = nnscore
                    bestpval = significant_reducers_pval[jq]
                    

        logger.debug(f'Nearest neighbor perturbation of {target} is {nnbr}')

        ranking = sorted(ranking, key=lambda x: x[1], reverse=True)

        logger.debug(f'Nearest neighbor ranking: {ranking}')


        #We have the neighboring perturbation, now we find the effect of this perturbation on each cell type and then apply the corresponding to each cell in the heldout data.

        nnbr_pert = nnbr


        #Mean control state of each cell type


        if self.mean_shift:


            selected_cell_control_mean = self.train_adata[(self.train_adata.obs[CELL_KEY] == cell_id) & (self.train_adata.obs[PERT_KEY] == CONTROL_PERT)].obsm['embedding'].mean(axis=0)

            selected_cell_nbr_pert_mean = self.train_adata[(self.train_adata.obs[CELL_KEY] == cell_id) & (self.train_adata.obs[PERT_KEY] == nnbr_pert)].obsm['embedding'].mean(axis=0)
            
        
            pert_effect = selected_cell_nbr_pert_mean - selected_cell_control_mean

            predictions = adata.copy()

    
            predictions.obsm['embedding'] = pert_effect + predictions.obsm['embedding']


            return predictions.obsm['embedding']
        
        else:
            logger.debug(f"Running substitution method")
            adata_nbr = self.train_adata[(self.train_adata.obs[CELL_KEY] == cell_id) & (self.train_adata.obs[PERT_KEY] == nnbr_pert)]
            logger.debug(f"Number of cells with cell_id {cell_id} and perturbation {nnbr_pert} in training data {len(adata_nbr)}")
            res = adata_nbr #sc.pp.subsample(adata_nbr, n_obs=len(adata), replace=True, copy=True)
            return res.obsm['embedding']

        

    


