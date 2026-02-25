FROM python:3.9-slim

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

# ---- Install remaining deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -r requirements.txt

# ---- Copy Files ----
COPY src/ /app/src/
COPY reference_parameters.yaml /app/
COPY models/ /app/models
COPY ic/ /app/ic

# # ---- Sanity check ----
# RUN python - <<EOF
# import torch
# import torch_geometric
# import yaml, sklearn, matplotlib, pygame
# from solver import Solver
# print("All critical imports OK")
# EOF

EXPOSE 8000

CMD ["uvicorn", "solver_api:app", "--host", "0.0.0.0", "--port", "8000"]
