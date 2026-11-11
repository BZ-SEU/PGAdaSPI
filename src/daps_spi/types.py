"""State containers shared by physics, sampling and selection."""
from dataclasses import dataclass, field
import torch
from .io import digest

@dataclass
class PatternBatch:
    continuous: torch.Tensor  # [b,d], signed DMD coordinates [-1,1]
    plus: torch.Tensor  # [b,d], uint8
    minus: torch.Tensor
    rows: torch.Tensor  # [b,d], effective signed pattern / sqrt(d)
    ids: list[str]
    constraints: dict = field(default_factory=dict)


@dataclass
class MeasurementState:
    A: torch.Tensor  # [M,d], scaled effective rows
    y: torch.Tensor  # [M], scaled exposure differences
    plus: torch.Tensor  # [M,d], actual uint8 exposures; minus=255-plus
    pattern_ids: list[str]
    committed_actions: list[str] = field(default_factory=list)
    @property
    def M(self) -> int: return self.A.shape[0]
    @property
    def exposures(self) -> int: return 2*self.M
    @property
    def state_hash(self) -> str: return digest(self.A, self.y, self.plus, self.pattern_ids, 'all-pixels',
                                                'signed/sqrt(d);uint8-complement;no-noise')
    def append(self, batch: PatternBatch, y: torch.Tensor, action_id: str) -> 'MeasurementState':
        if action_id in self.committed_actions: raise ValueError('action already committed')
        return MeasurementState(torch.cat([self.A,batch.rows]),torch.cat([self.y,y]),
              torch.cat([self.plus,batch.plus]),self.pattern_ids+batch.ids,self.committed_actions+[action_id])


@dataclass
class ScenarioEnsemble:
    samples: torch.Tensor  # [L,d], physical domain; intentionally NOT clipped
    role: str
    seeds: list[int]
    sampling_config: dict
    checkpoint_hash: str
    state_hash: str
    diagnostics: dict = field(default_factory=dict)


@dataclass
class ActionDecision:
    batch: PatternBatch
    action_id: str
    state_hash: str
    basis: dict  # Only O design information and constraints; sealed before observation
