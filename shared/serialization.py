import io
import torch

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t.cpu(), buf)
    return buf.getvalue()

def bytes_to_tensor(b: bytes) -> torch.Tensor:
    buf = io.BytesIO(b)
    return torch.load(buf, map_location="cpu")
