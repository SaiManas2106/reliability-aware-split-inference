"""
Minimal, explicit split for torchvision ResNet-18.

We run forward in two phases:
- front(x, split_id) -> activation tensor
- back(activation, split_id) -> logits

Supported split_ids: stem, layer1, layer2, layer3, layer4
"""
from typing import Literal
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

SplitId = Literal["stem", "layer1", "layer2", "layer3", "layer4"]

class ResNet18Splitter(nn.Module):
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.eval()
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def front(self, x: torch.Tensor, split_id: SplitId) -> torch.Tensor:
        m = self.model
        x = x.to(self.device)

        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        if split_id == "stem":
            return x.detach().cpu()

        x = m.layer1(x)
        if split_id == "layer1":
            return x.detach().cpu()

        x = m.layer2(x)
        if split_id == "layer2":
            return x.detach().cpu()

        x = m.layer3(x)
        if split_id == "layer3":
            return x.detach().cpu()

        x = m.layer4(x)
        if split_id == "layer4":
            return x.detach().cpu()

        raise ValueError(f"Unsupported split_id: {split_id}")

    @torch.no_grad()
    def back(self, act: torch.Tensor, split_id: SplitId) -> torch.Tensor:
        m = self.model
        x = act.to(self.device)

        if split_id == "stem":
            x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        elif split_id == "layer1":
            x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        elif split_id == "layer2":
            x = m.layer3(x); x = m.layer4(x)
        elif split_id == "layer3":
            x = m.layer4(x)
        elif split_id == "layer4":
            pass
        else:
            raise ValueError(f"Unsupported split_id: {split_id}")

        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        x = m.fc(x)
        return x.detach().cpu()

    @torch.no_grad()
    def full(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = self.model(x)
        return y.detach().cpu()
