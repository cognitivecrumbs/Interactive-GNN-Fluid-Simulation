import sys
import os

import yaml
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)  # insert(0, ...) gives your path priority
from src.utils.initialization import load_in
from torch_geometric.data import Data
from interface import Interface

def test_interface():
    with open(os.path.join('src','testing','reference_parameters.yaml'),'r') as f:
        solver_inputs = yaml.load(f,Loader=yaml.FullLoader)

    pos,field,n,edges = load_in(solver_inputs)

    data = Data(x=field,n=n,edge_index=edges,q_0=pos).cpu()

    # intialize user interface and solver
    interface = Interface(solver_inputs,data)