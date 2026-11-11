"""Core frozen propagation-full-pseudoinverse pipeline."""
from pathlib import Path
import hashlib, json, os
import torch

def digest(*items) -> str:
    h = hashlib.sha256()
    for item in items:
        if isinstance(item, torch.Tensor):
            a = item.detach().cpu().contiguous()
            h.update(str((a.dtype, tuple(a.shape))).encode()); h.update(a.numpy().tobytes())
        else: h.update(json.dumps(item, sort_keys=True, default=str).encode())
    return h.hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def atomic_save(obj: object, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temp); os.replace(temp, path)
