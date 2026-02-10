# FROM python:3.9-slim

# # Avoid Python buffering issues
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1

# WORKDIR /src

# # System deps (torch + scientific libs)
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     git \
#     curl \
#  && rm -rf /var/lib/apt/lists/*

# # Install Python deps
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip \
#  && pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY src/ ./src/

# # Expose FastAPI port
# EXPOSE 8000

# # Run the server
# CMD ["uvicorn", "src.solver_api:src", "--host", "0.0.0.0", "--port", "8000"]


# FROM python:3.9-slim

# ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1

# WORKDIR /app

# # System deps (Torch + scientific stack)
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     git \
#     curl \
#  && rm -rf /var/lib/apt/lists/*

# # Install Python deps
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip \
#  && pip install --no-cache-dir -r requirements.txt

# # Copy source code
# COPY src/ ./src/

# # Make src importable
# ENV PYTHONPATH=/app/src

# EXPOSE 8000

# CMD ["uvicorn", "solver_api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.9-slim
# FROM python:3.9

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# System deps (needed for pygame, matplotlib, pyg)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# # ---- Install PyTorch first (CPU) ----
# RUN pip install torch==2.1.2 torchvision torchaudio \
#     --index-url https://download.pytorch.org/whl/cpu

# # ---- Install PyTorch Geometric wheels (MATCH TORCH VERSION) ----
# RUN pip install \
#     torch-scatter \
#     torch-sparse \
#     torch-cluster \
#     torch-spline-conv \
#     torch-geometric \
#     -f https://data.pyg.org/whl/torch-2.1.2+cpu.html

RUN pip install torch==2.0.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

# ---- Install remaining deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy source ----
COPY src/ /app/src/
COPY reference_parameters.yaml /app/
COPY models/ /app/models
COPY ic/ /app/ic

# ---- Sanity check ----
RUN python - <<EOF
import torch
import torch_geometric
import yaml, sklearn, matplotlib, pygame
from solver import Solver
print("All critical imports OK")
EOF

EXPOSE 8000

CMD ["uvicorn", "solver_api:app", "--host", "0.0.0.0", "--port", "8000"]
