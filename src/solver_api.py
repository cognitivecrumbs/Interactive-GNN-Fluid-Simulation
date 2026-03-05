from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from contextlib import asynccontextmanager
from torch_geometric.data import Data
from src.utils.initialization import load_in
from typing import List, Optional

import os
import sys
import yaml
import copy
import io
import torch

import asyncio

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, parent_dir)  
from solver import Solver


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("api_defaults/reference_parameters.yaml", "r") as f:
        solver_inputs = yaml.load(f, Loader=yaml.FullLoader)

    app.state.solver_inputs = solver_inputs
    app.state.solver        = {}
    app.state.locks         = {}

    yield

    # optional cleanup
    del app.state.solver


app = FastAPI(lifespan=lifespan)

# curl http://localhost:8000/predict/;
@app.get("/")
async def root():
    return {
        "Name": "Interactive GNN API",
        "description": "This is a api for predicting the next graph state using a pretrained model",
    }

# predict
# curl -X POST http://localhost:8000/predict/; echo
@app.post("/predict/")
async def predict(nsteps: int = 1):
    solver = app.state.solver['1']
    for i in range(nsteps):
        solver.update()
    z,_,_ = solver.pull_field()
    return {
        "Return": "Complete",
        "check": z.shape,
        "check1": z.numpy().tolist()
    }
# 

@app.get("/instance/")
async def instance_status():
    # return {
    #     "instance": list(app.state.solver.keys())
    # }
    status = {}
    required_fields = ['z','q_0','n','edge_index']
    for instance_id in app.state.solver.keys():
        missing_fields = []
        for check_field in required_fields:
            if not hasattr(app.state.solver[instance_id],check_field):
                missing_fields.append(check_field)

        if len(missing_fields) == 0:
            status[instance_id] = "ready to run"
        else:
            status[instance_id] = "Missing ,"
            for field in missing_fields:
                status[instance_id] += field + ", "
            status[instance_id] = status[instance_id][:-2]
            status[instance_id] += ". Initialize data"

    if len(status) == 0:
        return {"message": "No instances created"}

    return status

@app.post("/instance/{instance_id}")
async def create_instance(instance_id:str,
                          input_yaml:Optional[UploadFile] = File(None),
                          model_parameters:Optional[UploadFile] = File(None),
                          fm_model_parameters:Optional[UploadFile] = File(None)):
    app.state.solver[instance_id] = Solver(
        app.state.solver_inputs if input_yaml is None else yaml.load(io.BytesIO(await input_yaml.read()), Loader=yaml.FullLoader),
        model_parameters = None if model_parameters is None else torch.load(io.BytesIO(await model_parameters.read()), map_location="cpu"),
        fm_parameters    = None if fm_model_parameters is None else torch.load(io.BytesIO(await fm_model_parameters.read()), map_location="cpu")
    )

    # setup locking while predictions
    app.state.locks[instance_id] = asyncio.Lock()
    return {'message': 'instance created',
            'instance_id': instance_id}

@app.post("/instance/delete/{instance_id}")
async def delete_instance(instance_id:str):
    del app.state.solver[instance_id]

    # setup locking while predictions
    del app.state.locks[instance_id]
    return {'message': 'instance deleted',
            'instance_id': instance_id}

@app.post("/{instance_id}/initialize/")
async def initialize(instance_id: str,
                     pos: UploadFile = File(...),
                     field: UploadFile = File(...),
                     n: UploadFile = File(...),
                     edge: UploadFile = File(...)
                    ):
    
    if instance_id not in app.state.solver:
        raise HTTPException(status_code=404, detail="instance not found")
    
    pos_t   = torch.load(io.BytesIO(await pos.read()),   map_location="cpu").float()
    field_t = torch.load(io.BytesIO(await field.read()), map_location="cpu").float()
    n_t     = torch.load(io.BytesIO(await n.read()),     map_location="cpu").long()
    edge_t  = torch.load(io.BytesIO(await edge.read()),  map_location="cpu").long()

    if len(field_t.shape) > 2:
        field_t = field_t[0]

    async with app.state.locks[instance_id]:
        app.state.solver[instance_id].update_state(q_0=pos_t,z=field_t,n=n_t,edge_index=edge_t)

    return {
        "status": "initialized",
        "instance_id": instance_id
    }

@app.post("/{instance_id}/update_state/")
async def update_state(
    instance_id: str,
    pos: Optional[UploadFile] = File(None),
    field: Optional[UploadFile] = File(None),
    n: Optional[UploadFile] = File(None),
    edge: Optional[UploadFile] = File(None)
):
    if instance_id not in app.state.solver:
        raise HTTPException(status_code=404, detail="instance not found")

    if pos is None and field is None and n is None and edge is None:
        raise HTTPException(
            status_code=400,
            detail="At least one state component must be provided"
        )

    pos_t = (
        torch.load(io.BytesIO(await pos.read()), map_location="cpu").float()
        if pos is not None else None
    )

    field_t = (
        torch.load(io.BytesIO(await field.read()), map_location="cpu").float()
        if field is not None else None
    )

    n_t = (
        torch.load(io.BytesIO(await n.read()), map_location="cpu").long()
        if n is not None else None
    )

    edge_t = (
        torch.load(io.BytesIO(await edge.read()), map_location="cpu").long()
        if edge is not None else None
    )

    # normalize field shape if needed
    if field_t is not None and field_t.ndim > 2:
        field_t = field_t[0]

    async with app.state.locks[instance_id]:
        app.state.solver[instance_id].update_state(
            q_0=pos_t,
            z=field_t,
            n=n_t,
            edge_index=edge_t
        )

    return {
        "status": "updated",
        "instance_id": instance_id
    }




# curl -X POST http://localhost:8000/instance_id/predict/?nsteps=1; echo
@app.post("/{instance_id}/predict/")
async def predict(instance_id:str, inner_steps: int = 1, outer_steps: int = 1):
    if instance_id not in app.state.solver:
        raise HTTPException(status_code=404, detail="instance not found")
    # create history storage
    z_hist = torch.empty(outer_steps,*app.state.solver[instance_id].z.shape)

    solver = app.state.solver[instance_id]
    async with app.state.locks[instance_id]:
        for i in range(outer_steps):
            for j in range(inner_steps):
                solver.update()
            z,_,_ = solver.pull_field()
            z_hist[i] = z

    buffer = io.BytesIO()
    torch.save(z_hist.cpu(), buffer)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=field_hist.pt"
        }
    )

@app.post("/{instance_id}/gen_new_mesh/")
async def update_state(
    instance_id: str,
    outer_steps: int = 1,
    inner_steps: int = 100
):

    async with app.state.locks[instance_id]:
        for _ in range(outer_steps):
            app.state.solver[instance_id].update_mesh(
                n_steps=inner_steps
            )

    return {
        "status": "generated new mesh based on boundary conditions using flow matching model",
        "instance_id": instance_id
    }