from typing import Iterator, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import ResNet18_Weights

def get_cifar10_loader(
    root: str = "./data",
    train: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    limit: Optional[int] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Iterator over (x, y) where x is preprocessed for ResNet-18 ImageNet weights.
    CIFAR-10 is resized/normalized by the weights' transforms.
    """
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    ds = datasets.CIFAR10(root=root, train=train, download=True, transform=preprocess)
    if limit is not None:
        ds.data = ds.data[:limit]
        ds.targets = ds.targets[:limit]

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    for x, y in dl:
        yield x, y
