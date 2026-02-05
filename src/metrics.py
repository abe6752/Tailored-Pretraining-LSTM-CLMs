import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from src.model.generative_models.generative_models.utils import Smiles2RDKitCanonicalSmiles
from src.chem_utils import canonicalize_smiles

def gen_smiles_metrics(train_smiles: pd.DataFrame, gen_smiles: pd.DataFrame, trainsmi_col='smiles', gensmi_col='smiles'):
    gen_smiles = gen_smiles.copy()
    gen_smiles['canonical_smiles'] = gen_smiles[gensmi_col].apply(Smiles2RDKitCanonicalSmiles)
    valid_mols = gen_smiles['canonical_smiles'].dropna()
    nvalid     = len(valid_mols)
    unique 	   = valid_mols.drop_duplicates()

    # Novelty against canonicalized training SMILES
    train_cano_smi = canonicalize_smiles(train_smiles[trainsmi_col])
    train_set = set(train_cano_smi)
    isnovel	= ~unique.isin(train_set) if len(unique) > 0 else []
    novelty = isnovel.sum() / len(unique) if len(unique) > 0 else 0.0

    # Novelty against randomized training SMILES
    valid_idx = valid_mols.index
    uncano_valid  = gen_smiles.loc[valid_idx, gensmi_col].dropna()
    uncano_unique = uncano_valid.drop_duplicates()
    uncano_uniqueness = len(uncano_unique)/len(uncano_valid) if len(uncano_valid) > 0 else 0

    uncano_train_smi = train_smiles[trainsmi_col].dropna().unique().tolist()
    uncano_train_set = set(uncano_train_smi)
    if len(uncano_unique) > 0:
        uncano_isnovel   = ~uncano_unique.isin(uncano_train_set)
        uncano_novelty   = uncano_isnovel.sum() / len(uncano_unique)
    else:
        uncano_novelty = 0.0

    avg_length = np.mean([len(smi) for smi in unique]) if len(unique) > 0 else 0.0

    return dict(validity=float(nvalid)/len(gen_smiles) if len(gen_smiles) > 0 else 0.0, 
                unique=float(len(unique))/nvalid if nvalid > 0 else 0,
                unique_uncano=uncano_uniqueness,
                novelty=novelty,
                novelty_randomized_SMILES=uncano_novelty,
                avg_length=avg_length
                )


def cohens_d(x1, x2):
    nx1, nx2 = len(x1), len(x2)
    v1, v2   = np.var(x1, ddof=1), np.var(x2, ddof=1)  # np.var(x1, ddof=1): sample variance
    s = np.sqrt(((nx1 - 1) * v1 + (nx2 - 1) * v2) / (nx1 + nx2 - 2)) 
    d = np.abs(np.mean(x1) - np.mean(x2)) / s
    return d


def mean_NN_gen_to_ref(gen_fps, ref_fps):
    """
    Calculate the mean nearest neighbor Tanimoto similarity from generated molecules to reference molecules.
    
    For each generated molecule, finds the highest Tanimoto similarity to any reference molecule,
    then returns the mean of these maximum similarities.
    
    Args:
        gen_fps: List of RDKit fingerprints for generated molecules
        ref_fps: List of RDKit fingerprints for reference molecules
        
    Returns:
        float: Mean of the maximum Tanimoto similarities
    """
    nn_vals = np.full(len(gen_fps), -1.0, dtype=float)
    for ref_fp in ref_fps:
        sims = DataStructs.BulkTanimotoSimilarity(ref_fp, gen_fps)
        nn_vals = np.maximum(nn_vals, sims)
    return float(nn_vals.mean())


def calc_rediscovery(gen_smi: list, test_smi: list) -> float:
    """
    Calculate the rediscovery rate of test molecules in generated molecules.
    
    Measures what fraction of test molecules are found among the generated molecules.
    This metric evaluates how well the generative model can reproduce known molecules.
    
    Args:
        gen_smi: List of generated SMILES strings
        test_smi: List of test SMILES strings to check for rediscovery
        
    Returns:
        float: Rediscovery rate (fraction of test molecules found in generated set)
    """
    test_in_gen = set(test_smi) & set(gen_smi)
    rediscovery = len(test_in_gen) / len(test_smi)
    return rediscovery