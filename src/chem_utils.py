import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def is_valid_smiles(smi):
    return Chem.MolFromSmiles(smi) is not None

def canonicalize_smiles(smiles_iter):
    cano_list = []
    seen = set()
    for smi in smiles_iter:
        if not isinstance(smi, str) or not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        cano = Chem.MolToSmiles(mol)
        if cano not in seen:
            seen.add(cano)
            cano_list.append(cano)
    return cano_list


def read_smiles_list_tsv(path, col='canonical_smiles', dropna=False, dedup=False):
    """Extract SMILES from a TSV file, removing NaNs and duplicates.
    
    Args:
        path (str): Path to the TSV file
        col (str): Column name containing SMILES strings
        dropna (bool): Whether to remove rows with NaN values
        dedup (bool): Whether to remove duplicate SMILES
        
    Returns:
        list: List of SMILES strings
    """
    df = pd.read_table(path, index_col=0)
    if dropna:
        df = df.dropna(subset=[col])
    if dedup:
        df = df.drop_duplicates(subset=[col])
    return df[col].tolist()


def smiles_to_mols(smiles_iter, return_invalid=False):
    """Convert SMILES strings to RDKit molecule objects.
    
    Args:
        smiles_iter: Iterable of SMILES strings
        return_invalid (bool): Whether to return invalid SMILES as well
        
    Returns:
        list: List of RDKit molecule objects
        tuple: (mols, invalid) if return_invalid=True
    """
    mols     = []
    invalid  = []
    for smi in smiles_iter:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(smi)
        else:
            mols.append(mol)
    if return_invalid:
        return mols, invalid
    return mols


def calc_properties(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Calculate molecular properties for SMILES in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing SMILES strings
        smiles_col (str): Column name containing SMILES strings
        
    Returns:
        pd.DataFrame: DataFrame with added molecular properties (mw, logP, tpsa)
    """
    df_out = df.copy()
    df_out['mol'] = df_out[smiles_col].apply(Chem.MolFromSmiles)
    n_invald = df_out['mol'].isna().sum()
    if n_invald > 0:
        print(f'[calc_properties] {n_invald} invalid SMILES removed')
    df_out = df_out.dropna(subset='mol') # delete Invalid SMILES
    props = {
        'mw': Descriptors.MolWt,
        'logP': Descriptors.MolLogP,
        'tpsa': Descriptors.TPSA,
    }
    for name, func in props.items():
        df_out[name] = df_out['mol'].apply(func)

    return df_out


def smiles2scaffold(smiles: str, kekulize=True):
    """Extract Murcko scaffold from a SMILES string.
    
    Args:
        smiles (str): SMILES string
        kekulize (bool): Whether to return Kekule form of scaffold
        
    Returns:
        str: Scaffold SMILES string, or None if extraction fails
    """
    mol = Chem.MolFromSmiles(smiles)
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold_mol, kekuleSmiles=kekulize)
    except:
        scaffold_smiles = None
    return scaffold_smiles


def mols_to_morgan_fps(mols, radius=2, n_bits=1024):
    MORGAN_GEN = GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [MORGAN_GEN.GetFingerprint(mol) for mol in mols]


def generate_self_tanimoto_matrix(smiles_list, r=2, bit=1024):

    mols = smiles_to_mols(smiles_list)

    fps = mols_to_morgan_fps(mols, radius=r, n_bits=bit)

    n   = len(fps)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        mat[i][i] = 1.0
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for j, s in enumerate(sims, start=i + 1):
            mat[i][j] = mat[j][i] = s

    df = pd.DataFrame(mat)

    return df
