import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam

from shared.data_covertype import get_covertype_train_test
from shared.split_mlp_covertype import MLPClassifier

def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits, y) * bs
            n += bs
    return total_loss / n, total_acc / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--out", type=str, default="artifacts/mlp_covertype.pt")
    args = ap.parse_args()

    device = torch.device("cpu")
    train_loader, test_loader, _ = get_covertype_train_test(
        batch_size_train=args.batch_size,
        batch_size_test=2048,
    )

    model = MLPClassifier().to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"epoch={epoch} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model_state": model.state_dict(), "best_test_acc": best_acc}, out_path)

    print(f"Saved best checkpoint to {out_path} with test_acc={best_acc:.4f}")

if __name__ == "__main__":
    main()
