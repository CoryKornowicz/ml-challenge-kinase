

from pathlib import Path
from rdkit import Chem 
from rdkit import RDConfig, Chem
from rdkit.Chem import AllChem
from torch_geometric.data.storage import GlobalStorage
import networkx as nx 
import numpy as np 
import torch
import torch.nn as nn 

feature_factory = AllChem.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
# Pharmacophoric Feature Definition List 
FEATURE_DEFS = list(sorted({key.split(".")[0] for key in feature_factory.GetFeatureDefs().keys()}))
# Standard atom type map 
ATOM_TYPES = {"C": 6, "N": 7, "O": 8, "Si": 14, "S": 16, "F": 9, "P": 15, "Cl": 17, "Br": 35, "I": 53}

def count_parameters(model: nn.Module):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def one_hot_from_lst(val, lst) -> np.ndarray:
    """Convert a value to a one-hot vector."""
    one_hot = [0]*len(lst)
    one_hot[lst.index(val)] = 1
    return one_hot


def mol_to_nx(mol: Chem.Mol) -> nx.Graph:
    """Generate a networkx graph from a RDKit molecule.

    Args:
        mol (Chem.Mol): Chem.Mol object

    Returns:
        nx.Graph: networkx graph with nodes as atoms and edges as bonds, positions are included as well
    """
    G = nx.Graph()
    mol_features = feature_factory.GetFeaturesForMol(mol)
    atomCoords = mol.GetConformer().GetPositions()

    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1: continue
        
        atom_fp = [0]*len(FEATURE_DEFS)
        if atom_features := [feat for feat in mol_features if atom.GetIdx() in feat.GetAtomIds()]:
            for feat in atom_features:
                atom_fp[FEATURE_DEFS.index(feat.GetFamily())] = 1 
        
        # Build feature vector 
        feat_vect = np.array(one_hot_from_lst(atom.GetAtomicNum(), list(ATOM_TYPES.values()) + atom_fp), dtype=float)
        
        G.add_node(atom.GetIdx(),
                   pos = atomCoords[i],
                   x=feat_vect)
        
    for bond in mol.GetBonds():
        if mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 1: continue
        if mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 1: continue
        
        isConjugated = 1 if bond.GetIsConjugated() and not bond.GetIsAromatic() else 0
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   edge_attr=np.array(one_hot_from_lst(bond.GetBondTypeAsDouble(),[1.0, 1.5, 2.0, 3.0, 99.0]) + [isConjugated], dtype=float))
    return G


def convert_storagevalue(store: GlobalStorage, data_type: type):
    """Covert all data stores inside Batch object to the same data type

    Args:
        store (GlobalStorage): Batch or Data object
        data_type (type): torch data type to convert to 
    """
    for key in store.keys:
        # check if the key is not meant to be an index  
        if store[key].dtype != data_type and key not in ["batch", "edge_index"]:
            store[key] = store[key].to(data_type)
            

# Averaging meter for tracking loss progress
class AverageMeter: 
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        if isinstance(val, torch.Tensor | np.ndarray | list):
            self.val = np.mean(val)
        else:
            self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return str(self.avg)