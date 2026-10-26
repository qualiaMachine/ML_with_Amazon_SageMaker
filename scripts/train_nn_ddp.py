"""Train TitanicNet with data parallelism: one training run split across instances.

The companion of train_nn.py in the "Training part 2" episode. Same network, same
data, same full-batch loop, plus the three things data parallelism needs:

1. join the process group the launcher set up: one copy of this script per
   instance, each numbered from zero. A copy's number is its RANK; the number
   of copies is the WORLD SIZE. The rank is the only thing that differs between
   the copies, so everything that must differ between instances keys off it;
2. give each copy its own shard of the training rows, chosen by its rank;
3. wrap the model in DistributedDataParallel, which averages the gradients
   across copies after every backward pass, so every copy of the model takes
   the same step and stays identical.

Launched by SageMaker with distribution={"mpi": {"enabled": True,
"processes_per_host": 1}} it runs on CPU or GPU instances; launched with
torchrun it works too; run plainly (python train_nn_ddp.py ...) it trains on
one process like train_nn.py. Only rank 0 prints the metrics and saves the model.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel


class TitanicNet(nn.Module):
    """The same network as train_nn.py: 7 features -> 64 -> 32 -> 1 probability of survival."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def setup_distributed():
    """Join the process group if a launcher started several copies of this script; return (rank, world_size).

    A launcher is whatever started the copies: SageMaker's MPI launcher
    (distribution={"mpi": ...} on the estimator) or torchrun. Each tells the
    copies who they are through environment variables, but with different
    names. PyTorch's own rendezvous (init_method="env://") reads four of them:
    RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT. This function fills those
    four in from whatever the launcher provided, then joins. With no launcher
    at all, it reports rank 0 of a world of 1 and nothing distributed happens.
    """
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        # Started by SageMaker's MPI launcher. MPI names the copies with its own variables:
        # OMPI_COMM_WORLD_RANK is this copy's number, OMPI_COMM_WORLD_SIZE the number of copies.
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        # Copy them to the names PyTorch's rendezvous reads.
        os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world_size)
        # The copies have to agree on one machine and port to meet at. SageMaker lists the instances'
        # hostnames in SM_HOSTS (a JSON list such as ["algo-1", "algo-2"]); the first one hosts the meeting.
        # setdefault keeps a value the launcher already set, and the fallback makes a local test run work.
        os.environ.setdefault("MASTER_ADDR", json.loads(os.environ.get("SM_HOSTS", '["localhost"]'))[0])
        os.environ.setdefault("MASTER_PORT", "7777")   # any free port, the same on every instance
    elif "WORLD_SIZE" in os.environ:
        # Started by torchrun, which sets RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT itself.
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    else:
        return 0, 1   # no launcher: a plain `python train_nn_ddp.py`, one copy, nothing to join
    if world_size == 1:
        return 0, 1   # a launcher started a single copy: joining a group of one is pointless
    # The backend is the library the copies exchange gradients with: nccl is NVIDIA's, GPU to GPU;
    # gloo talks over ordinary TCP and is what CPU instances use.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # Join: every copy calls this and blocks until all world_size copies have arrived at MASTER_ADDR:MASTER_PORT.
    dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size


def calculate_accuracy(preds, labels):
    return (torch.round(preds) == labels).float().mean()


def train(train_path, val_path, epochs, learning_rate):
    rank, world_size = setup_distributed()   # 1. join the process group (or run alone with no launcher)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, val_data = np.load(train_path), np.load(val_path)
    X_train = torch.tensor(train_data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(train_data["y_train"], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(val_data["X_val"], dtype=torch.float32).to(device)
    y_val = torch.tensor(val_data["y_val"], dtype=torch.float32).unsqueeze(1).to(device)

    # 2. The shard: every world_size-th row starting at this copy's rank, so with two copies rank 0
    #    takes rows 0, 2, 4, ... and rank 1 takes rows 1, 3, 5, ... Together the slices are the whole
    #    training set with no row seen twice, and no copy needed to ask another which rows to take.
    X_shard, y_shard = X_train[rank::world_size].to(device), y_train[rank::world_size].to(device)
    print(f"[rank {rank} of {world_size}] training on {len(X_shard)} of {len(X_train)} rows", flush=True)

    torch.manual_seed(0)   # the same initial weights on every process; DDP checks and keeps them in step
    model = TitanicNet().to(device)
    if world_size > 1:
        # 3. DDP: after every loss.backward(), the gradients are averaged across all processes
        #    before optimizer.step(), so each process takes the same step on the same model.
        model = DistributedDataParallel(model)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_shard), y_shard)
        loss.backward()      # with DDP, this is where the gradient exchange happens
        optimizer.step()

        if rank == 0 and ((epoch + 1) % 100 == 0 or epoch == epochs - 1):
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_accuracy = calculate_accuracy(val_outputs, y_val)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}", flush=True)
            print(f"validation:accuracy = {val_accuracy:.4f}", flush=True)

    if rank == 0:   # one saved model is enough: every copy holds the same weights, and rank 0 exists in every world
        weights = model.module.state_dict() if world_size > 1 else model.state_dict()
        model_output_path = "/opt/ml/model/nn_model.pth" if os.path.exists("/opt/ml/model") else "nn_model.pth"
        torch.save(weights, model_output_path)
        print(f"Model saved to {model_output_path}", flush=True)
    if world_size > 1:
        dist.barrier()   # let rank 0 finish saving before the other processes exit
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TitanicNet with data parallelism")
    parser.add_argument("--train", type=str, required=True, help="Path to training data (.npz file)")
    parser.add_argument("--val", type=str, required=True, help="Path to validation data (.npz file)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()
    train(args.train, args.val, args.epochs, args.learning_rate)
