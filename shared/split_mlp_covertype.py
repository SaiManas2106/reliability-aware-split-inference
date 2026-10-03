from typing import Literal
from pathlib import Path
import torch
import torch.nn as nn

SplitId = Literal["h1", "h2", "h3"]

class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int = 54, num_classes: int = 7):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return x

class MLPCovertypeSplitter(nn.Module):
    def __init__(self, checkpoint_path: str | Path, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.model = MLPClassifier()
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def front(self, x: torch.Tensor, split_id: SplitId) -> torch.Tensor:
        x = x.to(self.device)
        x = self.model.relu(self.model.fc1(x))
        if split_id == "h1":
            return x.detach().cpu()
        x = self.model.relu(self.model.fc2(x))
        if split_id == "h2":
            return x.detach().cpu()
        x = self.model.relu(self.model.fc3(x))
        if split_id == "h3":
            return x.detach().cpu()
        raise ValueError(f"Unknown split_id: {split_id}")

    @torch.no_grad()
    def back(self, act: torch.Tensor, split_id: SplitId) -> torch.Tensor:
        x = act.to(self.device)
        if split_id == "h1":
            x = self.model.relu(self.model.fc2(x))
            x = self.model.relu(self.model.fc3(x))
            x = self.model.out(x)
            return x.detach().cpu()
        if split_id == "h2":
            x = self.model.relu(self.model.fc3(x))
            x = self.model.out(x)
            return x.detach().cpu()
        if split_id == "h3":
            x = self.model.out(x)
            return x.detach().cpu()
        raise ValueError(f"Unknown split_id: {split_id}")

    @torch.no_grad()
    def full(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = self.model(x)
        return y.detach().cpu()
