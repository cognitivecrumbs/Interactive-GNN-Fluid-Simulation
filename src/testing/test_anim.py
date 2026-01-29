import sys
import os

import yaml
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)  
import pytest
import torch

from src.utils.initialization import load_in
from torch_geometric.data import Data
from src.solver import Solver
from src.utils.anim_utils import *
def test_generate_hist():
    iters = 10
    with open(os.path.join('src','testing','reference_parameters.yaml'),'r') as f:
        solver_inputs = yaml.load(f,Loader=yaml.FullLoader)

    pos,field,n,edges = load_in(solver_inputs)

    # data = Data(x=field.detach().clone(),n=n.detach().clone(),edge_index=edges,q_0=pos).cpu()
    data = Data(x=field,n=n,edge_index=edges,q_0=pos).cpu()
    solver = Solver(solver_inputs,data)

    # create a test net
    # class TestModel(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()

    #     def forward(self, z, n, edge_index, q_0):
    #         # z = z.detach()
    #         # z += 0.1
    #         return torch.ones_like(z)*0.1
        
    # solver.net = TestModel().to(solver.device)

    field_history,pos_history,n_history = generate_histories(None,iters,None,solver)

    assert len(field_history) == iters
    assert len(pos_history)   == iters
    assert len(n_history)     == iters

    solver.update_state(field,pos,edges,n)

    for i in range(iters):
        # if solver.dims['d'] == 0:
        #     z = z0 + solver.dt*(0.1*i)
        # else:
        #     derivitive = z0[:,:-solver.dims['d']] + solver.dt*(0.1*i)
        #     direct = (0.1*i)[:,-solver.dims['d']:]
        #     z = torch.cat([derivitive,direct],dim=1)[:]

        # # denormalize
        # z = z*solver.stats_z['std'] + solver.stats_z['mean']
        # # apply bcs

        solver.update()
        solver.update_bc()
        field_reference = solver.pull_field()[0]

        assert field_history[i].cpu().numpy() == pytest.approx(field_reference.cpu().numpy())
        assert pos_history[i].cpu().numpy() == pytest.approx(pos)
        assert n_history[i].cpu().numpy() == pytest.approx(n)

def test_animate_hist(tmp_path):
    with open(os.path.join('src','testing','reference_anim_parameters.yaml'),'r') as f:
        anim_inputs = yaml.load(f,Loader=yaml.FullLoader)
    output_file = tmp_path / (anim_inputs['save_name'] + '.gif')
    anim_inputs['save_dir'] = tmp_path
    iters = 10
    anim_inputs['iter'] = iters-1

    with open(os.path.join('src','testing','reference_parameters.yaml'),'r') as f:
        solver_inputs = yaml.load(f,Loader=yaml.FullLoader)

    pos,field,n,edges = load_in(solver_inputs)

    # data = Data(x=field.detach().clone(),n=n.detach().clone(),edge_index=edges,q_0=pos).cpu()
    data = Data(x=field,n=n,edge_index=edges,q_0=pos).cpu()
    solver = Solver(solver_inputs,data)

    # create a test net
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, z, n, edge_index, q_0):
            # z = z.detach()
            # z += 0.1
            return torch.ones_like(z)*0.1
        
    solver.net = TestModel().to(solver.device)

    field_history,pos_history,n_history = generate_histories(None,iters,None,solver)

    animate_history(anim_inputs,solver_inputs,field_history,pos_history,n_history,[edges]*iters)
    with open(os.path.join('src','testing','reference_animation.gif'),'rb') as f:
        reference_content = f.read()
    with open(output_file, 'rb') as f:
        actual_content = f.read()
        assert reference_content == actual_content

