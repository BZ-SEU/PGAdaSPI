"""Exact affine projection; stable QR/SVD fallback when the Gram matrix loses rank."""
import torch

class AffineProjector:
    def __init__(self, A: torch.Tensor, y: torch.Tensor, rtol: float=1e-10):
        self.A=A.double(); self.y=y.double(); self.rtol=rtol
        values,vectors=torch.linalg.eigh(self.A@self.A.T)
        threshold=rtol*values.abs().max(); keep=values>threshold
        self.rank=int(keep.sum()); self.spectrum=values
        self.vectors=vectors[:,keep]; self.values=values[keep]
        self.qr_basis=None
        if self.rank < len(self.y):
            # Gram eigenvalues square singular values. Its conservative threshold
            # must not discard valid rows accepted by the full pseudoinverse.
            # Preserve the established well-conditioned path; use QR/SVD on A
            # itself for sensitive states, with the same singular-value rtol.
            q,r=torch.linalg.qr(self.A.T,mode='reduced')
            u,s,vh=torch.linalg.svd(r,full_matrices=False)
            active=s>rtol*s.max()
            self.rank=int(active.sum())
            self.qr_basis=q@u[:,active]
            self.left_vectors=vh[active].T
            self.singular_values=s[active]
            incompatible=self.y-self.left_vectors@(self.left_vectors.T@self.y)
            if incompatible.norm()>1e-7*(1+self.y.norm()):
                raise ValueError('inconsistent rank-deficient noiseless measurements (QR/SVD)')
            return
        incompatible=self.y-self.vectors@(self.vectors.T@self.y)
        if incompatible.norm()>1e-7*(1+self.y.norm()): raise ValueError('inconsistent rank-deficient noiseless measurements')
    def __call__(self, m: torch.Tensor) -> torch.Tensor:
        """m [...,d], exact affine projection; no damping and no pixel-space projector."""
        residual=self.y-m.double()@self.A.T
        if self.qr_basis is not None:
            return m.double()+((residual@self.left_vectors)/self.singular_values)@self.qr_basis.T
        correction=((residual@self.vectors)/self.values)@self.vectors.T@self.A
        return m.double()+correction
