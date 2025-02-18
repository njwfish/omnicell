from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import yaml
import json
import copy
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass()
class ModelConfig:
    """Configuration for the model."""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ModelConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
            name = data.pop('name')
            return cls(name=name, parameters=data)


@dataclass()
class ETLConfig:        
    """Configuration for ETL process."""
    name: str
    count_norm: bool = False
    log1p: bool = False
    drop_unmatched_perts: bool = False
    HVG: bool = False
    synthetic_model_config_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> 'ETLConfig':
        with open(path) as f:
            return cls(**yaml.safe_load(f))

@dataclass()
class EmbeddingConfig:
    """Configuration for all embeddings attached to the the dataset."""
    gene_embedding: Optional[str] = None
    pert_embedding: Optional[str] = None
    metric_space: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> 'EmbeddingConfig':
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    @property
    def name (self) -> str:
        """Get the name of the embedding configuration."""

        #format: gemb_X_pemb_Y_mspace_Z
        name_parts = []
        if self.gene_embedding:
            name_parts.append(f"gemb_{self.gene_embedding}")
        if self.pert_embedding:
            name_parts.append(f"pemb_{self.pert_embedding}")
        if self.metric_space:
            name_parts.append(f"mspace_{self.metric_space}")

        return '_'.join(name_parts)

@dataclass()
class DatasplitConfig:
    """Configuration for data splitting."""
    name: str
    dataset: str
    mode: str
    holdout_cells: List[str] = field(default_factory=list)
    holdout_perts: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> 'DatasplitConfig':
        with open(path) as f:
            return cls(**yaml.safe_load(f))

@dataclass()
class EvalConfig:
    """Configuration for evaluation."""
    name: str
    dataset: str
    evaluation_targets: List[Tuple[str, str]]

    @classmethod
    def from_yaml(cls, path: str) -> 'EvalConfig':
        with open(path) as f:
            return cls(**yaml.safe_load(f))

@dataclass()     
class Config:
    """Configuration for a model or task."""
    model_config: ModelConfig
    etl_config: ETLConfig
    datasplit_config: DatasplitConfig
    embedding_config: Optional[EmbeddingConfig] = None
    eval_config: Optional[EvalConfig] = None



    @classmethod
    def from_yamls(cls, 
                  model_yaml: str,
                  etl_yaml: str, 
                  datasplit_yaml: str,
                  embed_yaml: Optional[str] = None,
                  eval_yaml: Optional[str] = None) -> 'Config':
        """Load config from separate YAML files."""
        config = cls(
            model_config=ModelConfig.from_yaml(model_yaml),
            etl_config=ETLConfig.from_yaml(etl_yaml),
            datasplit_config=DatasplitConfig.from_yaml(datasplit_yaml),
            embedding_config=EmbeddingConfig.from_yaml(embed_yaml) if embed_yaml else None,
            eval_config=EvalConfig.from_yaml(eval_yaml) if eval_yaml else None
        )
        logger.info(f"Loaded config: {config}")
        return config

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        return cls(
            model_config=ModelConfig(**config_dict['model_config']),
            etl_config=ETLConfig(**config_dict['etl_config']),
            datasplit_config=DatasplitConfig(**config_dict['datasplit_config']),
            embedding_config=EmbeddingConfig(**config_dict['embedding_config']) if 'embedding_config' in config_dict else None,
            eval_config=EvalConfig(**config_dict['eval_config']) if 'eval_config' in config_dict else None
        )

    def to_dict(self) -> dict:
        config_dict = {
            'model_config': self.model_config.__dict__,
            'etl_config': self.etl_config.__dict__,
            'datasplit_config': self.datasplit_config.__dict__,
        }
        
        if self.embedding_config is not None:
            config_dict['embedding_config'] = self.embedding_config.__dict__
            
        if self.eval_config is not None:
            config_dict['eval_config'] = self.eval_config.__dict__
            
        return config_dict
    


    def get_train_hash(self) -> str:
        """Get hash of training configuration."""
        return hashlib.sha256(json.dumps(self.get_training_config().to_dict(), sort_keys=True).encode()).hexdigest()[:8]

    # [Rest of the methods remain the same, but use the new structured config objects]
    def get_train_path(self) -> Path:
        """Get the path for training artifacts."""
        datasplit_prefix = self._get_datasplit_prefix()
        datasplit_prefix_path = f"{datasplit_prefix}/" if datasplit_prefix else ""

        
        return Path(
            f"./models/{self.datasplit_config.dataset}"
            f"{f'/{self.embedding_config.name}' if self.embedding_config is not None else ''}"
            f"/{self.etl_config.name}"
            f"/{self.model_config.name}"
            f"/{datasplit_prefix_path}{self.datasplit_config.name}"
            f"/{self.get_train_hash()}"
        ).resolve()


    def get_synthetic_config_ID(self) -> str:
        """Get a unique, human readable ID of the config used to generate the synthetic data associated with this config."""

        synthetic_config_name = f"{self.model_config.name}_{self.etl_config.name}"

        #We select relevant parts of the config:
        synthetic_config = copy.deepcopy(self)
        
        
        #Only keep model, ETL (no synthetic), and datasplit
        synthetic_config.eval_config = None
        synthetic_config.embedding_config = None
        synthetic_config.etl_config.synthetic = None

        hash_synthetic_data_config = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:8]

        return f"{synthetic_config_name}_{hash_synthetic_data_config}"

    def get_eval_path(self) -> Path:
        """Get the path for evaluation artifacts."""
        datasplit_prefix = self._get_datasplit_prefix()
        datasplit_prefix_path = f"{datasplit_prefix}/" if datasplit_prefix else ""
        
        train_and_eval_hash = hashlib.sha256(
            f"{self.get_train_hash()}/{self.get_eval_hash()}".encode()
        ).hexdigest()[:8]
        
        return Path(
            f"./results/{self.datasplit_config.dataset}"
            f"{f'/{self.embedding_config.name}' if self.embedding_config is not None else ''}"
            f"/{self.etl_config.name}"
            f"/{self.model_config.name}"
            f"/{datasplit_prefix_path}{self.datasplit_config.name}"
            f"/{train_and_eval_hash}"
        ).resolve()

    def _get_datasplit_prefix(self) -> Optional[str]:
        """Helper method to get datasplit prefix."""
        datasplit_name_parts = self.datasplit_config.name.split('-')
        return '-'.join(datasplit_name_parts[0:-1]) if len(datasplit_name_parts) > 1 else None


    def get_eval_hash(self) -> str:
        """Get hash of evaluation configuration."""
        if not self.eval_config:
            raise ValueError("No evaluation config present")
        return hashlib.sha256(
            json.dumps(self.eval_config.__dict__).encode()
        ).hexdigest()[:8]

    def get_training_config(self) -> 'Config':
        """Get a new Config instance with only training-related configurations."""
        return Config(
            embedding_config=self.embedding_config,
            model_config=self.model_config,
            etl_config=self.etl_config,
            datasplit_config=self.datasplit_config
        )