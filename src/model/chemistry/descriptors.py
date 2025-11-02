from rdkit import Chem

def HeavyAtomCount(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f'cannot conver smiles {smi} to Mol. Return None')
        return None
    
    return mol.GetNumAtoms()