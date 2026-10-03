from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def _prepare(test_size: float = 0.2, random_state: int = 42):
    X, y = fetch_covtype(return_X_y=True, as_frame=False)
    y = y.astype(np.int64) - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test, scaler

def get_covertype_train_test(
    batch_size_train: int = 1024,
    batch_size_test: int = 1,
    test_limit: Optional[int] = None,
    random_state: int = 42,
):
    X_train, X_test, y_train, y_test, scaler = _prepare(random_state=random_state)
    if test_limit is not None:
        X_test = X_test[:test_limit]
        y_test = y_test[:test_limit]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader, scaler

def get_covertype_test_loader(limit: Optional[int] = None, batch_size: int = 1, random_state: int = 42):
    _, test_loader, _ = get_covertype_train_test(
        batch_size_train=1024,
        batch_size_test=batch_size,
        test_limit=limit,
        random_state=random_state,
    )
    return test_loader
