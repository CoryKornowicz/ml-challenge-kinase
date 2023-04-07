

from flask import Flask, render_template, request, redirect, url_for, render_template_string, jsonify
from rdkit.Chem.AllChem import Compute2DCoords
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
import numpy as np
import os
import sys 
import torch
from typing import Tuple

sys.path.append(os.path.abspath("."))

from kinpred.dataset import MolEntry
from kinpred.gnn_models import LigandModel
from kinpred.utils import mol_to_nx

## Setup flask app that display the index page as the default page
app = Flask(__name__)

## Assistance functions and model wrapper definition

class Model():
    """Model Class to hold our GNN model and make predictions
    """
    def __init__(self):
        self.wrapped_model = LigandModel(in_channels=18,edge_features=6,hidden_dim=128,residual_layers=4,mlp_layers=3,\
                    key_dim=1024,dropout_rate=0.15).double()
        self.wrapped_model.load_state_dict(torch.load(os.path.abspath("kinpred/weights/GNN/gnn_model.pth"), map_location=torch.device('cpu')))
        self.wrapped_model.eval()
    
    def predict(self, mol: MolEntry) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the pIC50 values and kinase classes of an input molecule

        Args:
            mol (MolEntry): MolEntry object to predict against

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of pIC50 and kinase class predictions as numpy arrays. They are reshaped into column vectors for easy concatenation to target labels
        """
        mol_graph = Batch.from_data_list([from_networkx(mol_to_nx(mol.to_mol()))])
        fingerprint = torch.from_numpy(np.array([mol.to_fp()]))
        outenergy, outcls = self.wrapped_model(mol_graph, fingerprint)
        out_energy = outenergy.detach().numpy().round(3).reshape(1,-1).T
        out_cls = torch.sigmoid(outcls).detach().numpy().round(3).reshape(1,-1).T
        return out_energy, out_cls
        
# Instantiate the model in the global scope to save time on repetitive predictions
model = Model()

## Target map for folding api prediction results into JSON response
TARGET_MAP = {"JAK1": 0,
              "JAK2": 1,
              "JAK3": 2,
              "TYK2": 3}

# Convenience function to draw a molecule and return the HTML svg tag
def draw(mol: MolEntry):
    #Create canvas 
    drawer = rdMolDraw2D.MolDraw2DSVG(200, 118)
    # Make 2D coords for cleaner viewing 
    Compute2DCoords(mol.to_mol())
    # Draw
    drawer.DrawMolecule(mol.to_mol())
    # Committ buffer
    drawer.FinishDrawing()
    #Return svg text to be rendered in HTML template
    return drawer.GetDrawingText()


# FLASK FUNCTIONS

#Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Main prediciton page which returns the rendered HTML including the molecule image(s) and prediction(s)
@app.route('/predict', methods = ['POST'])
def predict():
    if request.method != 'POST':
        print('Invalid request type', file=sys.stderr)
        return redirect(url_for('index'))
    res = []
    # Test for file input
    if len(request.files.keys()) > 0:
        f = request.files['file']
        # assert file is .smi 
        if f.filename.split('.')[-1] != 'smi':
            print('File must be .smi', file=sys.stderr)
            return redirect(url_for('index'))
        # Grab all smiles from the file 
        # print()
        mol_smiles = []
        # with open(f.filename, 'r') as file:
        file_data = f.stream.read().decode('utf-8')
        for line in file_data.splitlines(keepends=True):
            mol_smiles.append(line)
        # Predict and draw each molecule
        for smi in mol_smiles:
            molecule = MolEntry(smiles=smi)
            out_energy, out_cls = model.predict(molecule)
            draw_res = draw(molecule)
            outpred = np.hstack((out_energy, out_cls))
            res.append([draw_res, outpred, smi])
    
    # Test for smiles input
    elif len(request.form.keys()) > 0:
        smi = request.form["smiles"]
        molecule = MolEntry(smiles=smi)
        out_energy, out_cls = model.predict(molecule)
        draw_res = draw(molecule)
        outpred = np.hstack((out_energy, out_cls))
        res = [[draw_res, outpred, smi]]
    else:
        # No input detected
        print('No file or smiles provided', file=sys.stderr)
        return redirect(url_for('index'))
        
    return render_template('mlview.html', results=res)


# Define API endpoint which returns no rendered HTML
@app.route('/api_predict', methods = ['POST'])
def api_predict():
    """Method to predict the pIC50 values and kinase classes of an input SMILES, currently only supports single SMILES input

    Returns:
        Union[int|dict]: Returns either -1 for error or a dictionary of predictions if successful
    """
    if request.method != 'POST':
        print('Invalid request type', file=sys.stderr)
        return -1

    if (data:=request.form["smiles"]) is None:
        print("No smiles provided", file=sys.stderr)
        return -1

    molecule = MolEntry(smiles=data)
    out_energy, out_cls = model.predict(molecule)
    outpred = np.hstack((out_energy, out_cls))
    tar_map = np.array(list(TARGET_MAP.keys())).reshape(1,-1).T
    outpred = np.hstack((tar_map, outpred))
    out_dict = {row[0]: {"Energy": row[1], "Prob": row[2]} for row in outpred}

    return jsonify(out_dict)
    

# Launch the app
if __name__ == '__main__':
    app.run()
