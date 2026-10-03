from typing import Literal
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

SplitId = Literal["features_0","features_3","features_6","features_10","features_13","features_16","features_18"]

_SPLIT_TO_LAST_IDX = {
    "features_0": 0,
    "features_3": 3,
    "features_6": 6,
    "features_10": 10,
    "features_13": 13,
    "features_16": 16,
    "features_18": 18,
}

class MobileNetV2Splitter(nn.Module):
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        self.model = mobilenet_v2(weights=weights)
        self.model.eval()
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def front(self, x: torch.Tensor, split_id: SplitId) -> torch.Tensor:
        x = x.to(self.device)
        last_idx = _SPLIT_TO_LAST_IDX[split_id]
        for idx in range(last_idx + 1):
            x = self.model.features[idx](x)
        return x.detach().cpu()

    @torch.no_grad()
    def back(self, act: torch.Tensor, split_id: SplitId) -> torch.Tensor:
        x = act.to(self.device)
        last_idx = _SPLIT_TO_LAST_IDX[split_id]
        for idx in range(last_idx + 1, len(self.model.features)):
            x = self.model.features[idx](x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x.detach().cpu()

    @torch.no_grad()
    def full(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        y = self.model(x)
        return y.detach().cpu()
