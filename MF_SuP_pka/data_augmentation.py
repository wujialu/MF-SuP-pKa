from rdkit import Chem
from rdkit.Chem import AllChem
from MF_SuP_pka.ionization_group import get_ionization_aid

def acid_to_base(smiles):
    a2b_smiles = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        mol_h = AllChem.AddHs(mol)
        acid_idx, base_idx = get_ionization_aid(mol_h)
        assert len(base_idx) == 0 and len(acid_idx) == 1
        atom = mol.GetAtomWithIdx(acid_idx[0])
        atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() - 1)
        a2b_smiles.append(Chem.MolToSmiles(mol))
    return a2b_smiles

def base_to_acid(smiles):
    b2a_smiles = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        mol_h = AllChem.AddHs(mol)
        acid_idx, base_idx = get_ionization_aid(mol_h)
        assert len(acid_idx) == 0 and len(base_idx) == 1
        atom = mol.GetAtomWithIdx(base_idx[0])
        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
        b2a_smiles.append(Chem.MolToSmiles(mol))
    return b2a_smiles