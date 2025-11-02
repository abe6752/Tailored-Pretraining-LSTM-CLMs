from pathlib import Path
import os

# Fine-tuning datasets
FINETUNE_FILTER = 'filtered'  #  'unfiltered'

def repo_root() -> Path:
    '''Return the repository root directory (2 levels up from this file)'''
    return Path(__file__).resolve().parents[1]

# Definition of directories as strings for easy concatenation
_repo_root_str = str(repo_root())
DATA_DIR         = os.path.join(_repo_root_str, 'data')
PRETRAIN_DATA    = os.path.join(DATA_DIR, 'pretrain')
FINETUNE_DATA    = os.path.join(DATA_DIR, 'finetune', FINETUNE_FILTER)

RESULTS_DIR      = os.path.join(_repo_root_str, 'results')
PRETRAIN_RESULTS = os.path.join(RESULTS_DIR, 'pretrain')
FINETUNE_RESULTS = os.path.join(RESULTS_DIR, 'finetune', FINETUNE_FILTER)

OUTPUT_DIR       = os.path.join(_repo_root_str, 'output')
PRETRAIN_OUT     = os.path.join(OUTPUT_DIR, 'pretrain')
FINETUNE_OUT     = os.path.join(OUTPUT_DIR, 'finetune', FINETUNE_FILTER)
FIGURES_DIR      = os.path.join(OUTPUT_DIR, 'figures')

def ensure_dirs():
    for path_str in [PRETRAIN_DATA, FINETUNE_DATA, PRETRAIN_RESULTS, 
                     FINETUNE_RESULTS, PRETRAIN_OUT, FINETUNE_OUT, FIGURES_DIR]:
        Path(path_str).mkdir(parents=True, exist_ok=True)