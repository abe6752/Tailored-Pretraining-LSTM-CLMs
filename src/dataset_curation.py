import os
import itertools
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from src.alert import GetStructuralAlerts

def save_smiles_list(smiles_list, smi_col, out_path):
    os.makedirs(out_path, exist_ok=True)
    pd.DataFrame(({smi_col: smiles_list})).to_csv(out_path, sep='\t', index=False)


def SmilesToRandomSmiles(smiles_series, num, seed=42):
    mol_list           = [Chem.MolFromSmiles(smiles) for smiles in smiles_series.to_list()]
    rd_smiles_list     = [rdmolfiles.MolToRandomSmilesVect(mol, num, randomSeed=seed) for mol in mol_list]
    combined_rd_smiles_list = list(itertools.chain(*rd_smiles_list))
    uniq_rdsmiles_list = list(set(combined_rd_smiles_list))
    return uniq_rdsmiles_list


def filter_structural_alerts(df, smiles_col):
    df = df.copy()
    df['mol'] = df[smiles_col].apply(Chem.MolFromSmiles)
    valid_mask = df['mol'].notna()

    filter_flags = GetStructuralAlerts(df.loc[valid_mask, 'mol'])
    filter_flags.insert(0, smiles_col, df.loc[valid_mask, smiles_col].values)

    flag_cols   = [c for c in filter_flags.columns if c != smiles_col]
    passed_mask = (filter_flags[flag_cols] == False).all(axis=1)
    passed_df   = filter_flags.loc[passed_mask, [smiles_col]].reset_index(drop=True)

    pass_rate = float(passed_mask.mean()) if len(passed_mask) else 0.0

    return passed_df, pass_rate, filter_flags