"""Exact expected MSE reduction of a candidate batch under the centered pseudoinverse."""
import torch
from .propagation_pinv import PseudoinverseDecoder

def gains(decoder: PseudoinverseDecoder, B: torch.Tensor, scenarios: torch.Tensor) -> torch.Tensor:
    """Per-scenario gain [L]: ||e||^2-||e-P_B e||^2 over d, e=scenario-xhat, P_B the update projector."""
    e=scenarios.double()-decoder.xhat
    K=decoder.update_map(B)
    u=(e@B.double().T)@K.T
    return (2*(e*u).sum(-1)-u.square().sum(-1))/e.shape[-1]
