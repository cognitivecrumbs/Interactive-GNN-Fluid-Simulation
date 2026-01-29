import sys
import os

import yaml
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)  # insert(0, ...) gives your path priority
import pytest
import torch
import numpy as np
from solver import Solver

from torch_geometric.data import Data

from src.utils.initialization import load_in

class TestSolver:

    def test_declaration(self):
        with open(os.path.join('src','testing','reference_parameters.yaml'),'r') as f:
            solver_inputs = yaml.load(f,Loader=yaml.FullLoader)

        pos,field,n,edges = load_in(solver_inputs)
        data = Data(x=field,n=n,edge_index=edges,q_0=pos).cpu()
        solver = Solver(solver_inputs,data)

        assert isinstance(solver,Solver)

        # pytorch stores values as float32
        assert solver.stats_z['mean'].cpu().numpy() == pytest.approx(np.array([1.37000167e+00, -5.79022290e-03, 2.30710119e-01],dtype=np.float32))
        assert solver.stats_z['std'].cpu().numpy() == pytest.approx(np.array([8.11523557e-01, 2.94183582e-01, 7.43897438e-01],dtype=np.float32))
        assert solver.stats_q['mean'].cpu().numpy() == pytest.approx(np.array([-2.90865205e-12, 8.88754782e-12,],dtype=np.float32))
        assert solver.stats_q['std'].cpu().numpy() == pytest.approx(np.array([1.95575710e-02, 1.90107096e-02],dtype=np.float32))

        assert isinstance(solver.net, torch.nn.Module)

        assert isinstance(solver.fm_net, torch.nn.Module)

        assert solver.dt == pytest.approx(0.01)

        assert isinstance(solver.z, torch.Tensor)
        assert isinstance(solver.q_0, torch.Tensor)
        assert isinstance(solver.n, torch.Tensor)

    def test_device_assignment(self):
        with open(os.path.join('src','testing','reference_parameters.yaml'),'r') as f:
            solver_inputs = yaml.load(f,Loader=yaml.FullLoader)

        solver_inputs['gpu'] = True
        solver = Solver(solver_inputs)
        if torch.cuda.is_available():
            assert solver.device == 'cuda'
        elif torch.backends.mps.is_available():
            assert solver.device == 'mps'
        else:
            assert solver.device == 'cpu'

        solver_inputs['gpu'] = False
        solver = Solver(solver_inputs)
        assert solver.device == 'cpu'

    def test_update_state(self):
        with open(os.path.join('src','testing','reference_parameters.yaml'),'r') as f:
            solver_inputs = yaml.load(f,Loader=yaml.FullLoader)
        pos,field,n,edges = load_in(solver_inputs)
        data = Data(x=field,n=n,edge_index=edges,q_0=pos).cpu()
        solver = Solver(solver_inputs)

        solver.update_state(field,pos,edges,n)

        assert isinstance(solver.z, torch.Tensor)
        assert isinstance(solver.q_0, torch.Tensor)
        assert isinstance(solver.n, torch.Tensor)
        assert isinstance(solver.edge_index, torch.Tensor)


    def test_update(self):
        with open(os.path.join('src','testing','reference_parameters.yaml'),'r') as f:
            solver_inputs = yaml.load(f,Loader=yaml.FullLoader)
        pos,field,n,edges = load_in(solver_inputs)
        data = Data(x=field,n=n,edge_index=edges,q_0=pos).cpu()
        solver = Solver(solver_inputs,data)

        # replace model with simple model for test
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, z, n, edge_index, q_0):
                # z = z.detach()
                # z += 0.1
                return z + 0.1
        
        solver.net = TestModel().to(solver.device)

        z_before          = solver.z.clone()
        q_0_before        = solver.q_0.clone()
        n_before          = solver.n.clone()
        edge_index_before = solver.edge_index.clone()

        solver.update()

        # assert torch.allclose(z_before + 0.1, solver.z)
        # assert torch.allclose(q_0_before,solver.q_0)
        # assert torch.allclose(n_before,solver.n)
        # assert torch.allclose(edge_index_before,solver.edge_index)
        if solver.dims['d'] == 0:
            z = z_before + solver.dt*(0.1+z_before)
        else:
            derivitive = z[:,:-solver.dims['d']] + solver.dt*(0.1+z_before[:,:-solver.dims['d']])
            direct = (0.1+z_before)[:,-solver.dims['d']:]
            z = torch.cat([derivitive,direct],dim=1)[:]
        assert solver.net_out.cpu().numpy() == pytest.approx(z_before.cpu().numpy() + 0.1)
        assert z.cpu().numpy()    == pytest.approx(solver.z.cpu().numpy())
        assert q_0_before.cpu().numpy()        == pytest.approx(solver.q_0.cpu().numpy())
        assert n_before.cpu().numpy()          == pytest.approx(solver.n.cpu().numpy())
        assert edge_index_before.cpu().numpy() == pytest.approx(solver.edge_index.cpu().numpy())

        # apply bcs
        solver.update_bc()

        # inlet test
        assert (solver.z[solver.n[:,1]==1,0]*solver.stats_z['std'][0]+solver.stats_z['mean'][0]).cpu().numpy() == pytest.approx(solver_inputs['inlet_vel'])
        assert (solver.z[solver.n[:,1]==1,1]*solver.stats_z['std'][1]+solver.stats_z['mean'][1]).cpu().numpy() == pytest.approx(0)
        # wall test
        assert (solver.z[solver.n[:,1]==3,0]*solver.stats_z['std'][0]+solver.stats_z['mean'][0]).cpu().numpy() == pytest.approx(0)
        assert (solver.z[solver.n[:,1]==3,1]*solver.stats_z['std'][1]+solver.stats_z['mean'][1]).cpu().numpy() == pytest.approx(0)
        # outlet test
        assert (solver.z[solver.n[:,1]==2,2]*solver.stats_z['std'][2]+solver.stats_z['mean'][2]).cpu().numpy() == pytest.approx(solver_inputs['outlet_p'])

        z,E,S = solver.pull_field()
        assert z[:,0].cpu().numpy() == pytest.approx((solver.z[:,0]*solver.stats_z['std'][0]+solver.stats_z['mean'][0]).cpu().numpy())
        assert z[:,1].cpu().numpy() == pytest.approx((solver.z[:,1]*solver.stats_z['std'][1]+solver.stats_z['mean'][1]).cpu().numpy())
        assert z[:,2].cpu().numpy() == pytest.approx((solver.z[:,2]*solver.stats_z['std'][2]+solver.stats_z['mean'][2]).cpu().numpy())
    
    def test_redistribute_nodes(self):
        with open(os.path.join('src','testing','reference_parameters.yaml'),'r') as f:
            solver_inputs = yaml.load(f,Loader=yaml.FullLoader)
        pos,field,n,edges = load_in(solver_inputs)
        data = Data(x=field,n=n,edge_index=edges,q_0=pos).cpu()
        solver = Solver(solver_inputs,data)

        pos_before = solver.q_0.clone()

        # simple torch model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, q_0, n, t, edge_index):
                return torch.ones_like(q_0)*0.1
            
        solver.fm_net = TestModel().to(solver.device)

        new_positions = solver.redistribute_nodes(solver.q_0,solver.n,n_steps=10)

        assert new_positions[solver.n[:,0] == 1].cpu().numpy() == pytest.approx(pos_before[solver.n[:,0] == 1].cpu().numpy() + 0.1)
        # assert not np.allclose(pos_before.cpu().numpy(),solver.q_0.cpu().numpy())

    def test_find_boundaries(self):
        with open(os.path.join('src','testing','reference_parameters.yaml'),'r') as f:
            solver_inputs = yaml.load(f,Loader=yaml.FullLoader)
        solver = Solver(solver_inputs)

        # create simple box in box with linspace points
        outer_box_top = [torch.linspace(0.01,0.99,20), torch.ones(20)]
        outer_box_bot = [torch.linspace(0.01,0.99,20), torch.zeros(20)]
        outer_box_lft = [torch.zeros(20), torch.linspace(0,1,20)]
        outer_box_rgh = [torch.ones(20), torch.linspace(0,1,20)]
        outer_box = torch.cat([torch.stack(outer_box_top,dim=1),torch.stack(outer_box_bot,dim=1),torch.stack(outer_box_lft,dim=1),torch.stack(outer_box_rgh,dim=1)],dim=0)
        # outer_box = np.unique(outer_box,axis=0)

        inner_box_top = [torch.linspace(0.4+0.01,0.6-0.01,20), torch.ones(20)*0.6]
        inner_box_bot = [torch.linspace(0.4+0.01,0.6-0.01,20), torch.ones(20)*0.4]
        inner_box_lft = [torch.ones(20)*0.4, torch.linspace(0.4,0.6,20)]
        inner_box_rgh = [torch.ones(20)*0.6, torch.linspace(0.4,0.6,20)]
        inner_box = torch.cat([torch.stack(inner_box_top,dim=1),torch.stack(inner_box_bot,dim=1),torch.stack(inner_box_lft,dim=1),torch.stack(inner_box_rgh,dim=1)],dim=0)
        # inner_box = np.unique(inner_box,axis=0)
 
        fill_points = torch.rand((100,2))
        fill_points = fill_points[(fill_points[:,0]<0.4) | (fill_points[:,0]>0.6) | (fill_points[:,1]<0.4) | (fill_points[:,1]>0.6)]

        all_points = torch.cat([outer_box,inner_box,fill_points],dim=0)
        n = np.zeros((all_points.shape[0],4),dtype=int)
        n[:-fill_points.shape[0],3] = 1
        n[-fill_points.shape[0]:,0] = 1

        reference_n = np.zeros((all_points.shape[0],3),dtype=int)
        reference_n[:outer_box.shape[0],1] = 1
        reference_n[outer_box.shape[0]:outer_box.shape[0]+inner_box.shape[0],2] = 1
        reference_n[-fill_points.shape[0]:,0] = 1
        fm_n = solver.find_boundaries(torch.tensor(all_points),torch.tensor(n))

        # print(reference_n)
        # print(fm_n)
        assert fm_n.cpu().numpy() == pytest.approx(reference_n)
