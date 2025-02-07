import os
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Constants
PERT_KEY = 'pert'
CELL_KEY = 'cell'
CONTROL_PERT = 'ctrl'
GENE_VAR_KEY = 'gene'
GENE_EMBEDDING_KEY = 'gene_embedding'
PERT_EMBEDDING_KEY = 'pert_embedding'

# Set root path - either from env var or current directory
if 'OMNICELL_ROOT' in os.environ:
    OMNICELL_ROOT = Path(os.environ['OMNICELL_ROOT'])
    logger.info(f"Using OMNICELL_ROOT from environment: {OMNICELL_ROOT}")
else:
    OMNICELL_ROOT = Path('/orcd/data/omarabu/001/njwfish/omnicell')
    logger.info(f"OMNICELL_ROOT not set, using current directory: {OMNICELL_ROOT.absolute()}")

# Derived paths
DATA_CATALOGUE_PATH = OMNICELL_ROOT / 'configs' / 'catalogue'

def get_data_path(relative_path: str) -> Path:
    """Get a path relative to OMNICELL_ROOT."""
    return OMNICELL_ROOT / relative_path

# Create necessary directories
logger.info(f"Data catalogue path: {DATA_CATALOGUE_PATH.absolute()}")