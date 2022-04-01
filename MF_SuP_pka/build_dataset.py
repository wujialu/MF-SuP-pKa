from dgl import DGLGraph
import pandas as pd
from rdkit.Chem import MolFromSmiles
import numpy as np
from dgl.data.graph_serialize import save_graphs, load_graphs, load_labels
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import random
from ionization_group import get_ionization_aid
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.MolStandardize import rdMolStandardize


def find_var(df, min_value):
    df1 = df.copy()
    df1.loc["var"] = np.var(df1.values, axis=0)
    col = [x for i, x in enumerate(df1.columns) if df1.iat[-1, i] < min_value]
    return col


def find_sum_0(df):
    '''
    input: df
    return: the columns of labels with no positive labels
    '''
    df1 = df.copy()
    df1.loc["sum"] = np.sum(df1.values, axis=0)
    col = [x for i, x in enumerate(df1.columns) if df1.iat[-1, i] == 0]
    return col


def find_sum_1(df):
    '''
    input: df
    return: the columns of labels with no negative labels
    '''
    df1 = df.copy()
    df1.loc["sum"] = np.sum(df1.values, axis=0)
    col = [x for i, x in enumerate(df1.columns) if df1.iat[-1, i] == len(df)]
    return col


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H=False, use_chirality=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
        ]) + one_of_k_encoding(atom.GetDegree(),
                               [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    # [0 if math.isnan(atom.GetDoubleProp('_GasteigerCharge')) or math.isinf(
    #     atom.GetDoubleProp('_GasteigerCharge')) else atom.GetDoubleProp('_GasteigerCharge')] +\
    # [0 if math.isnan(atom.GetDoubleProp('_GasteigerHCharge')) or math.isinf(
    #     atom.GetDoubleProp('_GasteigerHCharge')) else atom.GetDoubleProp('_GasteigerHCharge')]

    # [atom.GetIsAromatic()] # set all aromaticity feature blank.
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def one_of_k_atompair_encoding(x, allowable_set):
    for atompair in allowable_set:
        if x in atompair:
            x = atompair
            break
        else:
            if atompair == allowable_set[-1]:
                x = allowable_set[-1]
            else:
                continue
    return [x == s for s in allowable_set]


def bond_features(bond, use_chirality=True, atompair=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    if atompair:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        bond_feats = bond_feats + one_of_k_atompair_encoding(
            atom_pair_str, [['CC'], ['CN', 'NC'], ['ON', 'NO'], ['CO', 'OC'], ['CS', 'SC'],
                            ['SO', 'OS'], ['NN'], ['SN', 'NS'], ['CCl', 'ClC'], ['CF', 'FC'],
                            ['CBr', 'BrC'], ['others']]
        )
    return np.array(bond_feats).astype(float)


def etype_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats_1 = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
    ]
    for i, m in enumerate(bond_feats_1):
        if m == True:
            a = i

    bond_feats_2 = bond.GetIsConjugated()
    if bond_feats_2 == True:
        b = 1
    else:
        b = 0

    bond_feats_3 = bond.IsInRing
    if bond_feats_3 == True:
        c = 1
    else:
        c = 0

    index = a * 1 + b * 4 + c * 8
    if use_chirality:
        bond_feats_4 = one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
        for i, m in enumerate(bond_feats_4):
            if m == True:
                d = i
        index = index + d * 16
    return index


def construct_bigraph_from_smiles_and_idx(smiles, use_idx):
    """Construct a bi-directed DGLGraph with topology only for the molecule.

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.

    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    g = DGLGraph()

    # Add nodes
    mol = MolFromSmiles(smiles)
    if mol is not None:
        AllChem.ComputeGasteigerCharges(mol)
        Chem.AssignStereochemistry(mol)

        num_atoms = mol.GetNumAtoms()
        g.add_nodes(num_atoms)
        atom_feature_all = []

        node_centrality = [0] * num_atoms
        topo_adj_matrix = Chem.rdmolops.GetDistanceMatrix(mol)  # ndarray, (num_nodes, num_nodes)
        num_sg = len(use_idx)

        idx_sorted = sorted(use_idx)
        # print(idx, idx_sorted)
        use_adj = topo_adj_matrix[:, idx_sorted]  # (num_nodes, num_sg)
        use_adj = np.concatenate((use_adj, np.zeros((num_atoms, 10 - num_sg))), axis=1)  # (num_nodes,10)
        # print(use_adj)

        for atom_idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_feature = atom_features(atom)
            atom_feature_all.append(atom_feature)

            distance = []
            if atom_idx not in use_idx:
                for center in use_idx:
                    distance.append(len(Chem.rdmolops.GetShortestPath(mol, atom_idx, center)))
                node_centrality[atom_idx] = min(distance)

        g.ndata["node"] = torch.tensor(atom_feature_all)  # the original atom features
        g.ndata['centrality'] = torch.tensor(node_centrality)  # The distance from the nearest dissociation site
        g.ndata['use_adj'] = torch.tensor(use_adj)  # (num_nodes, 10)

        # Add edges
        src_list = []
        dst_list = []
        etype_feature_all = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            bond_feature = bond_features(bond)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            src_list.extend([u, v])
            dst_list.extend([v, u])
            etype_feature_all.append(bond_feature)
            etype_feature_all.append(bond_feature)

        g.add_edges(src_list, dst_list)

        g.edata["bond"] = torch.tensor(etype_feature_all)

        return g
    else:
        return 1


def construct_bigraph_from_smiles(smiles, acid_or_base=None):
    """Construct a bi-directed DGLGraph with topology only for the molecule.

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.

    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    g = DGLGraph()

    # Add nodes
    mol = MolFromSmiles(smiles)
    un = rdMolStandardize.Uncharger()
    mol = un.uncharge(mol)
    ionizable = [1]
    if mol is not None:
        num_atoms = mol.GetNumAtoms()
        g.add_nodes(num_atoms)
        atoms_feature_all = []
        acid_center = [0] * num_atoms
        base_center = [0] * num_atoms

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_feature = atom_features(atom)
            atoms_feature_all.append(atom_feature)
            if atom.GetSymbol() == 'N':
                if atom.GetTotalNumHs() > 0:
                    acid_center[i] = 1
                if atom.GetFormalCharge() <= 0:
                    base_center[i] = 1
            elif atom.GetSymbol() != 'C' and atom.GetTotalNumHs() > 0:
                acid_center[i] = 1

        if acid_or_base == 'acid':
            if acid_center == [0] * num_atoms:
                ionizable = [0]
        if acid_or_base == 'base':
            if base_center == [0] * num_atoms:
                ionizable = [0]

        g.ndata["node"] = torch.tensor(atoms_feature_all)
        g.ndata['acid'] = torch.tensor(acid_center)
        g.ndata['base'] = torch.tensor(base_center)

        # Add edges
        src_list = []
        dst_list = []
        etype_feature_all = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            bond_feature = bond_features(bond)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            src_list.extend([u, v])
            dst_list.extend([v, u])
            etype_feature_all.append(bond_feature)
            etype_feature_all.append(bond_feature)

        g.add_edges(src_list, dst_list)

        g.edata["bond"] = torch.tensor(etype_feature_all)
        return g
    else:
        return 1


def build_mask(labels_list, mask_value=100):
    mask = []
    for i in labels_list:
        if i == mask_value:
            mask.append(0)
        else:
            mask.append(1)
    return mask


def multi_task_build_dataset(dataset_smiles, labels_list, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_list]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        g_attentivefp = construct_attentivefp_bigraph_from_smiles(smiles)
        if g_attentivefp != 1:
            mask = build_mask(labels.loc[i], mask_value=123456)
            molecule = [smiles, g_attentivefp, labels.loc[i], mask, split_index.loc[i]]
            dataset_gnn.append(molecule)
            print('{}/{} molecule is transformed! {} is transformed failed!'.format(i + 1, molecule_number,
                                                                                    len(failed_molecule)))
        else:
            print('{} is transformed failed!'.format(smiles))
            molecule_number = molecule_number - 1
            failed_molecule.append(smiles)
    print('{}({}) is transformed failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn


def build_dataset(dataset_smiles, labels_list, smiles_name, acid_or_base=None):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_list]
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        if smiles == 123456:
            molecule_number = molecule_number - 1
            continue
        else:
            g_attentivefp = construct_bigraph_from_smiles(smiles, acid_or_base=acid_or_base)  # return g
            if g_attentivefp != 1:
                mask = build_mask(labels.loc[i], mask_value=123456)
                molecule = [smiles, g_attentivefp, labels.loc[i], mask]  # labels=pka value
                dataset_gnn.append(molecule)
                print('{}/{} molecule is transformed! {} is transformed failed!'.format(i + 1, molecule_number,
                                                                                        len(failed_molecule)))
            else:
                print('{} is transformed failed!'.format(smiles))
                molecule_number = molecule_number - 1
                failed_molecule.append(smiles)
    print('{}({}) is transformed failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn


def build_dataset_with_idx(dataset_smiles, labels_list, smiles_name, acid_or_base=None):
    dataset_gnn = []
    failed_molecule = []
    not_use_molecule = []
    labels = dataset_smiles[labels_list]
    smilesList = dataset_smiles[smiles_name]
    # idx = dataset_smiles[idx_name]
    molecule_number = len(smilesList)  # 123
    sg_id_ls = []
    for i, smiles in enumerate(smilesList):
        if smiles == 123456:
            molecule_number = molecule_number - 1
            continue
        else:
            mol = Chem.MolFromSmiles(smiles)
            # un = rdMolStandardize.Uncharger()
            # mol = un.uncharge(mol)
            # smiles = Chem.MolToSmiles(mol)
            mol_h = AllChem.AddHs(mol)

            acid_idx, base_idx = get_ionization_aid(mol_h)
            if acid_or_base == 'acid':
                use_idx = acid_idx
            elif acid_or_base == 'base':
                use_idx = base_idx
            else:
                print('please select acid or base to construct mol graphs')
                break

            if len(use_idx) > 0:
                g_attentivefp = construct_bigraph_from_smiles_and_idx(smiles, use_idx)  # return g
                if g_attentivefp != 1:
                    mask = build_mask(labels.loc[i], mask_value=123456)
                    num_sg = len(use_idx)
                    molecule = [smiles, g_attentivefp, labels.loc[i], mask, num_sg]  # labels=pka value
                    dataset_gnn.append(molecule)
                else:
                    print('{} is transformed failed!'.format(smiles))
                    molecule_number = molecule_number - 1
                    failed_molecule.append(smiles)
            else:
                print('{} does not have ionization group!'.format(smiles))
                molecule_number = molecule_number - 1
                not_use_molecule.append(smiles)
        print('{}/{} molecule is transformed! {} is transformed failed! {} does not have ionization group!'.format(
            i + 1, molecule_number, len(failed_molecule), len(not_use_molecule)))
    print('{}({}) is transformed failed!'.format(failed_molecule, len(failed_molecule)))
    print('{}({}) does not have ionization group!'.format(not_use_molecule, len(not_use_molecule)))
    return dataset_gnn


def pad_list(lst):
    inner_max_len = max(map(len, lst))
    map(lambda x: x.extend([123456] * (inner_max_len - len(x))), lst)
    return np.array(lst)


def built_data_and_save_for_pka(
        origin_path='data.csv',
        save_g_attentivefp_path='data_graph.bin',
        smiles_path='data_group.csv',
        task_list_selected=None,
        acid_or_base=None):
    data_origin = pd.read_csv(origin_path)
    data_origin = data_origin.fillna(123456)
    labels_list = [x for x in data_origin.columns if x not in ['smiles', 'SMILES', 'group', 'ChEMBL ID',
                                                               'num_acid_sites',
                                                               'num_base_sites']]  # pka_acidic or pka_basic
    if task_list_selected is not None:
        labels_list = task_list_selected
    smiles_name = 'smiles'
    data_set_gnn = build_dataset_with_idx(dataset_smiles=data_origin, labels_list=labels_list,
                                          smiles_name=smiles_name,
                                          acid_or_base=acid_or_base)
    smiles, g_attentivefp, labels, mask, num_sg = map(list, zip(*data_set_gnn))
    graph_labels = {'labels': torch.tensor(labels),
                    'mask': torch.tensor(mask),
                    'num_sg': torch.tensor(num_sg),
                    }


    smiles_pd = pd.DataFrame(smiles, columns=['smiles'])
    smiles_pd.to_csv(smiles_path, index=False)
    print('Molecules graph is saved!')
    save_graphs(save_g_attentivefp_path, g_attentivefp, graph_labels)


def standardization_np(data, mean, std):
    return (data - mean) / (std + 1e-10)


def re_standar_np(data, mean, std):
    return data * (std + 1e-10) + mean


def load_graph_from_csv_bin_for_pka_random_splited(
        bin_g_attentivefp_path='data_graph.bin',
        group_path='data_group.csv',
        shuffle=True):
    smiles = pd.read_csv(group_path, index_col=None).smiles.values

    g_attentivefp, detailed_information = load_graphs(bin_g_attentivefp_path)  # graph, graph_labels
    labels = detailed_information['labels']
    mask = detailed_information['mask']
    num_sg = detailed_information['num_sg']

    # calculate not_use index
    index_list = [x for x in range(len(mask))]
    if shuffle:
        random.shuffle(index_list)
    split_num = int(len(index_list) * 0.1)

    train_index = index_list[:8 * split_num]
    val_index = index_list[8 * split_num:9 * split_num]
    test_index = index_list[9 * split_num:]
    task_number = labels.size(1)
    train_set = []
    val_set = []
    test_set = []

    for i in train_index:
        molecule = [smiles[i], g_attentivefp[i], labels[i], mask[i], num_sg[i]]
        train_set.append(molecule)

    for i in val_index:
        molecule = [smiles[i], g_attentivefp[i], labels[i], mask[i], num_sg[i]]
        val_set.append(molecule)

    for i in test_index:
        molecule = [smiles[i], g_attentivefp[i], labels[i], mask[i], num_sg[i]]
        test_set.append(molecule)
    print(len(train_set), len(val_set), len(test_set), task_number)
    return train_set, val_set, test_set, task_number


def load_graph_from_csv_bin_for_pka_pretrain(
        bin_g_attentivefp_path='data_graph.bin',
        group_path='data_group.csv',
        shuffle=True):
    smiles = pd.read_csv(group_path, index_col=None).smiles.values

    g_attentivefp, detailed_information = load_graphs(bin_g_attentivefp_path)
    labels = detailed_information['labels']
    mask = detailed_information['mask']
    num_sg = detailed_information['num_sg']

    # calculate not_use index
    index_list = [x for x in range(len(mask))]
    if shuffle:
        random.shuffle(index_list)
    split_num = int(len(index_list) * 0.1)

    train_index = index_list[:8 * split_num]
    val_index = index_list[8 * split_num:]
    task_number = labels.size(1)
    train_set = []
    val_set = []
    for i in train_index:
        molecule = [smiles[i], g_attentivefp[i], labels[i], mask[i], num_sg[i]]
        train_set.append(molecule)

    for i in val_index:
        molecule = [smiles[i], g_attentivefp[i], labels[i], mask[i], num_sg[i]]
        val_set.append(molecule)

    print(len(train_set), len(val_set), task_number)
    return train_set, val_set, task_number


def load_graph_from_csv_bin_for_external_test(
        bin_g_attentivefp_path='data_graph.bin',
        group_path='data_group.csv'):
    smiles = pd.read_csv(group_path, index_col=None).smiles.values

    g_attentivefp, detailed_information = load_graphs(bin_g_attentivefp_path)  # graph, graph_labels
    labels = detailed_information['labels']
    mask = detailed_information['mask']
    num_sg = detailed_information['num_sg']

    # calculate not_use index
    index_list = [x for x in range(len(mask))]
    task_number = labels.size(1)

    test_set = []
    for i in index_list:
        molecule = [smiles[i], g_attentivefp[i], labels[i], mask[i], num_sg[i]]
        test_set.append(molecule)
    print(len(test_set), task_number)

    return test_set, task_number
