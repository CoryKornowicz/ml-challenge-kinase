

from .utils import mol_to_nx, convert_storagevalue
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Union, Optional
import networkx as nx
import numpy as np 
import pandas as pd
import torch


# Define Target Map to Index 
TARGET_MAP = {"JAK1": 0,
              "JAK2": 1,
              "JAK3": 2,
              "TYK2": 3}


# Dataclass representing underlying molecule object. Will be used to cache repeatedly used values, 
# i.e. finerprints, graphs, etc.
@dataclass
class MolEntry:
    smiles: str
    kinase_to_val: Optional[Dict[str, float]] = None
    mol: Optional[Chem.Mol] = None
    graph: Optional[Batch] = None
    fp: Optional[np.ndarray] = None
    
    # Convert SMILES to Morgan Fingerprint
    def to_fp(self, as_bits: bool = False):
        if self.fp is None:
            if self.mol is None: self.to_mol()
            fp = AllChem.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=1024)
            self.fp = fp
        if as_bits:
            return self.fp
        # convert to numpy array
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(self.fp, arr)
        return arr
    
    # Convert SMILES to Molcule Object
    def to_mol(self) -> Chem.Mol:
        if self.mol is None:
            self.mol = Chem.MolFromSmiles(self.smiles)
            self.mol = Chem.AddHs(self.mol)
            AllChem.EmbedMolecule(self.mol, randomSeed=0x1A77)
            ff = AllChem.UFFGetMoleculeForceField(self.mol)
            AllChem.OptimizeMolecule(ff, maxIters=500)
        return self.mol    


# Torch Dataset of MolEntry Objects
class KinPredDataset(Dataset):
    def __init__(self, entries: List[MolEntry]):
        self.entries = entries
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        return self.entries[idx]


# Torch DataLoader of MolEntry Objects for regressiion training that overrides collate_fn
class KinPredDataLoader(DataLoader):
    
    def __init__(self, dataset: KinPredDataset, batch_size: int, shuffle: bool, num_workers: int, target: str, device: str):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)
        self.target = target
        self.device = device or "cpu"
    
    def collate_fn(self, batch):
        fingerprints = np.array([entry.to_fp() for entry in batch])
        if self.target == "decoy":
            vals = np.zeros((len(batch), 1))
        else:
            vals = np.array([entry.kinase_to_val[self.target] for entry in batch]).reshape(-1, 1)

        # construct morgan fingerprints from smiles and then convert to tensors

        if self.device == "cpu":
            fingerprints = torch.from_numpy(fingerprints)
            vals = torch.from_numpy(vals)
        elif self.device == "mps":
            fingerprints = torch.from_numpy(fingerprints).float().to(self.device)
            vals = torch.from_numpy(vals).float().to(self.device)
        else:
            fingerprints = torch.from_numpy(fingerprints).to(self.device)
            vals = torch.from_numpy(vals).to(self.device)
        fingerprints.requires_grad = True
        vals.requires_grad = True
        return fingerprints, vals
    

def df_to_molentry_list(df: pd.DataFrame) -> Tuple[List[MolEntry],List[str]]:
    """Convert a dataframe into a list of MolEntry objects.

    Args:
        df: A pandas dataframe.

    Returns:
        A list of MolEntry objects.
    """
    # local store for holding molentry objects
    entries = {}
    for _, row in df.iterrows():
        if row['SMILES'] in entries:
            # Grab already instantiated MolEntry object and add the kinase value
            curmol = entries[row['SMILES']]
            curmol.kinase_to_val[row['Kinase_name']] = row['measurement_value']
        else:
            # Instantiate a new MolEntry object and add it to the local store
            entries[row['SMILES']] = MolEntry(smiles=row['SMILES'], kinase_to_val={row['Kinase_name']: row['measurement_value']})
    
    return list(entries.values()), df['Kinase_name'].unique()


def split_molentry_list(mol_list: List[MolEntry], test_size: float = 0.2) -> Tuple[KinPredDataset, KinPredDataset]:
    """Split a list of MolEntry objects into a train and test set.

    Args:
        mol_list: A list of MolEntry objects.
        test_size: The percentage of the list to use for the test set.

    Returns:
        A tuple of KinPredDataset objects [train|test].
    """
    train, test = train_test_split(mol_list, test_size=test_size)
    return KinPredDataset(train), KinPredDataset(test)


def split_mollist_to_target(mol_list: List[MolEntry], target: str, test_size: float = 0.2, include_negatives: bool = False) -> Union[Tuple[KinPredDataset, KinPredDataset], Tuple[KinPredDataset, KinPredDataset, KinPredDataset, KinPredDataset]]:
    """Sort a list of MolEntry objects into a train and test set against a specified target.

    Args:
        mol_list: A list of MolEntry objects.
        target: The target to filter by.
        test_size: The percentage of the list to use for the test set.
        include_negative: return negative samples as well.

    Returns:
        A tuple of KinPredDataset objects [train|test] with optionally negative samples included.
    """
    targeted = []
    nontargeted = []
    for entry in mol_list:
        if target in entry.kinase_to_val:
            targeted.append(entry)
        else:
            nontargeted.append(entry)
            
    targeted_train, targeted_test = train_test_split(targeted, test_size=test_size)
    nontargeted_train, nontargeted_test = train_test_split(nontargeted, test_size=test_size)
    
    if include_negatives:
        return KinPredDataset(targeted_train), KinPredDataset(targeted_test), KinPredDataset(nontargeted_train), KinPredDataset(nontargeted_test)
    
    return KinPredDataset(targeted_train), KinPredDataset(targeted_test)
    
    
def map_target_to_vals(mol: MolEntry) -> Tuple[np.ndarray, np.ndarray]:
        """Function to compose MolEntry into two labels, one-hot encoding and target-value. 
        
        Args:
            mol (MolEntry) 

        Returns:
            Tuple[np.ndarray, np.ndarray]: first is one-hot label, second is target pIC50
        """
        
        onehot = np.zeros((4,))
        pIC50 = np.zeros((4,))
        
        for target in mol.kinase_to_val.items():
            onehot[TARGET_MAP[target[0]]] = 1
            pIC50[TARGET_MAP[target[0]]] = target[1]
        
        return onehot, pIC50
        
## GRAPH CLASSES ## 

# Class designed for regression tasks specifically 
class KinPredGraphDataLoader(DataLoader):
    
    def __init__(self, dataset: KinPredDataset, batch_size: int, shuffle: bool, num_workers: int, target: str, device: str, include_fp: bool = False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)
        self.target = target
        self.device = device or "cpu"
        self.include_fp = include_fp
    
    # Convert a MolEntry object into a networkx graph which torch_geometric uses to make into a Data object
    def convert_mol_to_graph(self, mol: MolEntry):
        # Check if graph is already cached 
        if mol.graph is None:
            molgraph = from_networkx(mol_to_nx(mol.to_mol()))
            if self.device == "mps":
                convert_storagevalue(molgraph, torch.float32)
            mol.graph = molgraph
        return mol.graph
        

    def collate_fn(self, batch):
        # Transform batch into a Batch object of molecule graphs 
        batch_mols = Batch.from_data_list([self.convert_mol_to_graph(entry) for entry in batch])
        
        # If the target is decoy, we can only return 0 for the pIC50 value
        if self.target == "decoy":
            vals = np.zeros((len(batch), 1))
        else:
            vals = np.array([entry.kinase_to_val[self.target] for entry in batch]).reshape(-1, 1)

        # Clean up data for pytorch 

        if self.device == "cpu":
            vals = torch.from_numpy(vals).float()
        elif self.device == "mps":
            batch_mols = batch_mols.float().to(self.device)
            vals = torch.from_numpy(vals).float().to(self.device)
        else:
            batch_mols = batch_mols.to(self.device)
            vals = torch.from_numpy(vals).to(self.device)
        vals.requires_grad = True
        # If the fingerprint flag is set, return them as the last element of the tuple
        if self.include_fp:
            batch_fp = np.array([entry.to_fp() for entry in batch])
            if self.device == "mps":
                batch_fp = torch.from_numpy(batch_fp).float().to(self.device)
            else:
                batch_fp = torch.from_numpy(batch_fp).to(self.device)
            batch_fp.requires_grad = True
            return batch_mols, vals, batch_fp
        
        return batch_mols, vals
    
    
# Class designed with classification tasks in mind, returns one-hot encoded labels along with the predicted pIC50 value
class KinPredGraphClassDataLoader(DataLoader):

    def __init__(self, dataset: KinPredDataset, batch_size: int, shuffle: bool, num_workers: int, device: str, include_fp: bool = False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)
        self.device = device or "cpu"
        self.include_fp = include_fp
        
        
    def convert_mol_to_graph(self, mol: MolEntry):
        if mol.graph is None:
            molgraph = from_networkx(mol_to_nx(mol.to_mol()))
            if self.device == "mps":
                convert_storagevalue(molgraph, torch.float32)
            mol.graph = molgraph
        return mol.graph
        

    def collate_fn(self, batch):
        batch_mols = Batch.from_data_list([self.convert_mol_to_graph(entry) for entry in batch])
        
        # Map entries to return the one-hot encoded labels and target values
        labels, target_vals = zip(*[map_target_to_vals(entry) for entry in batch])
        labels = np.array(labels)
        target_vals = np.array(target_vals)

        # Clean up for pytorch 
        if self.device == "mps":
            batch_mols = batch_mols.float().to(self.device)
            labels = torch.from_numpy(labels).float().to(self.device)
            target_vals = torch.from_numpy(target_vals).float().to(self.device)
        else:
            batch_mols = batch_mols.to(self.device)
            labels = torch.from_numpy(labels).to(self.device)
            target_vals = torch.from_numpy(target_vals).to(self.device)
        
        labels.requires_grad = True
        target_vals.requires_grad = True
        
        # Fingerprint flag checking as well
        if self.include_fp:
            batch_fp = np.array([entry.to_fp() for entry in batch])
            if self.device == "mps":
                batch_fp = torch.from_numpy(batch_fp).float().to(self.device)
            else:
                batch_fp = torch.from_numpy(batch_fp).to(self.device)
            batch_fp.requires_grad = True
            return batch_mols, labels, target_vals, batch_fp
        
        return batch_mols, labels, target_vals