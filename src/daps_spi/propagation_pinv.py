"""Independent migration of the 2026-09-22 SPI_bound incoherent forward model.

No SPI_bound imports. Periodic, normalized |ASM impulse|²; scalar intensity
cascade. This is a simulation assumption, not a model validated for a laser rig.
"""
from dataclasses import dataclass, asdict
import math
import torch
from .types import MeasurementState


@dataclass(frozen=True)
class OpticalConfig:
    wavelength: float = 635e-9
    image_pitch: float = 10.8e-6
    dmd_pitch: float = 10.8e-6
    object_distance: float = .0225
    sensor_distance: float = .020
    sensor_width: float = .0097
    side: int = 256
    height: int | None = None
    width: int | None = None

    def __post_init__(self):
        # `side` remains the backward-compatible square shorthand.  New
        # templates may provide an explicit rectangular height/width.
        if self.height is None:
            object.__setattr__(self, 'height', self.side)
        if self.width is None:
            object.__setattr__(self, 'width', self.side)


class IntensityOptics:
    """Signed DMD codes -> effective object-intensity rows [B,d], scale 1/sqrt(d)."""
    def __init__(self, config: OpticalConfig, device: str = 'cpu'):
        self.config=config; self.height=int(config.height); self.width=int(config.width)
        self.side=self.height if self.height==self.width else None
        def spectrum(pitch, distance):
            # Reference explicitly constructs transfer in complex64 even for double input.
            fy=torch.fft.fftfreq(self.height,d=pitch,device=device)
            fx=torch.fft.fftfreq(self.width,d=pitch,device=device)
            ky,kx=torch.meshgrid(2*math.pi*fy,2*math.pi*fx,indexing='ij')
            kz=torch.sqrt(((2*math.pi/config.wavelength)**2-kx*kx-ky*ky).to(torch.complex64))
            transfer=torch.exp(1j*kz*distance)
            impulse=torch.fft.ifft2(transfer)
            psf=impulse.abs().square().double();psf=psf/psf.sum()
            return torch.fft.fft2(psf)
        self.h1=spectrum(config.image_pitch,config.object_distance)
        self.h2=spectrum(config.dmd_pitch,config.sensor_distance)
        ycoords=(torch.arange(self.height,device=device,dtype=torch.float64)-(self.height-1)/2)*config.dmd_pitch
        xcoords=(torch.arange(self.width,device=device,dtype=torch.float64)-(self.width-1)/2)*config.dmd_pitch
        self.mask=((ycoords[:,None].abs()<=config.sensor_width/2)&(xcoords[None,:].abs()<=config.sensor_width/2)).double()
        self.collection=self.leg(self.mask,self.h2,True)
        ypositions=ycoords/config.image_pitch+(self.height-1)/2
        xpositions=xcoords/config.image_pitch+(self.width-1)/2
        self.Ry=(1-(ypositions[:,None]-torch.arange(self.height,device=device,dtype=torch.float64)).abs()).clamp_min(0)
        self.Rx=(1-(xpositions[:,None]-torch.arange(self.width,device=device,dtype=torch.float64)).abs()).clamp_min(0)
        self.same_grid=config.image_pitch==config.dmd_pitch

    @staticmethod
    def leg(x: torch.Tensor, spectrum: torch.Tensor, adjoint: bool=False) -> torch.Tensor:
        return torch.fft.ifft2(torch.fft.fft2(x)*(spectrum.conj() if adjoint else spectrum)).real

    def effective_rows(self, signed: torch.Tensor) -> torch.Tensor:
        weights=signed.double().reshape(-1,self.height,self.width)*self.collection
        if not self.same_grid: weights=self.Ry.T@weights@self.Rx
        return self.leg(weights,self.h1,True).flatten(1)/math.sqrt(self.height*self.width)

    def observe(self, image: torch.Tensor, plus: torch.Tensor) -> torch.Tensor:
        """Independent full two-leg propagation of positive and negative exposures.

        image [H,W] intensity; codes [B,d] uint8 -> [B] differential pixel sums/sqrt(d).
        """
        illumination=self.leg(image.double().reshape(self.height,self.width),self.h1)
        if not self.same_grid: illumination=self.Ry@illumination@self.Rx.T
        results=[]
        for codes in plus.split(32):
            p=codes.double().reshape(-1,self.height,self.width)/255
            pos=(self.leg(illumination*p,self.h2)*self.mask).sum((-2,-1))
            neg=(self.leg(illumination*(1-p),self.h2)*self.mask).sum((-2,-1))
            results.append((pos-neg)/math.sqrt(self.height*self.width))
        return torch.cat(results)

    def full_scale(self) -> float:
        """Bucket reading of one all-on exposure of a uniform unit image (S_ref)."""
        white=torch.ones((self.height,self.width),dtype=torch.float64,device=self.h1.device)
        illumination=self.leg(white,self.h1)
        if not self.same_grid: illumination=self.Ry@illumination@self.Rx.T
        return float((self.leg(illumination,self.h2)*self.mask).sum())

    def metadata(self) -> dict:
        return dict(parameters=asdict(self.config),model='spatially incoherent intensity cascade',
                    boundary='circular FFT convolution; normalized intensity PSF; no padding',
                    kernel_precision='reference float32 frequencies / complex64 transfer and impulse; intensity promoted to float64 before normalization',
                    measurement='actual complementary uint8 exposures; difference / sqrt(d); sensor noise per settings.sensor_noise (none when sigma_rel=0)',
                    effective_row='H1^T R^T ((2*plus-255)/255 * H2^T sensor_mask) / sqrt(d)',
                    sensor_grid_coverage=float(self.mask.mean()),
                    limitations='incoherent DMD-plane assumption; not validated coherent laser optics; periodic wraparound; no absolute watts')


class SensorDarkNoise:
    """Additive zero-mean sensor dark-current noise on each physical exposure (simulator only).

    Each bucket reading S± becomes S± + n±, n± ~ N(0, sigma_d²), independent per exposure,
    sigma_d = sigma_rel * S_ref with S_ref = optics.full_scale(). The dark offset cancels exactly in
    the complementary difference (equal integration time), so only its fluctuation is simulated;
    y gains noise of std sqrt(2)*sigma_d/sqrt(d). Exposure k's noise depends only on
    (seed, image_index, k), never on the selected code, so methods sharing an image share noise and
    replays are exact. sigma_rel=0 returns exact zeros (noiseless path unchanged).
    """
    def __init__(self, config: dict | None, optics: IntensityOptics, seed: int, image_index: int):
        config=dict(config or {})
        self.model=config.get('model','dark_gaussian'); self.sigma_rel=float(config.get('sigma_rel',0.))
        if self.model!='dark_gaussian': raise ValueError(f'unknown sensor noise model {self.model}')
        if not math.isfinite(self.sigma_rel) or self.sigma_rel<0: raise ValueError('sensor_noise.sigma_rel must be finite and >= 0')
        self.d=optics.height*optics.width; self.full_scale=optics.full_scale()
        self.sigma_d=self.sigma_rel*self.full_scale; self.seed=int(seed); self.image_index=int(image_index)

    def __call__(self, start: int, count: int, device) -> torch.Tensor:
        """Differential noise [count] for effective rows start..start+count-1, already /sqrt(d)."""
        if self.sigma_d==0.: return torch.zeros(count,dtype=torch.float64,device=device)
        out=torch.empty(count,dtype=torch.float64)
        for i in range(count):
            g=torch.Generator().manual_seed(self.seed*1000003+self.image_index*100003+start+i)
            pos,neg=torch.randn(2,generator=g,dtype=torch.float64)*self.sigma_d
            out[i]=(pos-neg)/math.sqrt(self.d)
        return out.to(device)

    def metadata(self) -> dict:
        return dict(model=self.model,sigma_rel=self.sigma_rel,full_scale_S_ref=self.full_scale,sigma_dark_per_exposure=self.sigma_d,
                    sigma_y=math.sqrt(2)*self.sigma_d/math.sqrt(self.d),
                    definition='per exposure S±+n±, n±~N(0,(sigma_rel*S_ref)²) independent; dark offset cancels in the complementary difference; y noise std sqrt(2)*sigma_d/sqrt(d)',
                    seed_rule='torch.Generator seed = seed*1000003 + image_index*100003 + effective row index; independent of selected codes',
                    method_awareness='none: selection, projection, pinv and samplers are unchanged and treat y as exact')


class MeasurementFactor:
    """Shared fp64 thin QR of physical A [M,d]; no pixel-domain square matrix.

    Full row rank is required at the configured singular-value relative tolerance.
    Model projection uses z=2x-1 without storing A/2. Instances are state-bound.
    """
    def __init__(self, state: MeasurementState, rank_rtol: float=1e-10):
        self.full_dim=state.A.shape[1]
        self.A=state.A.double(); self.y=state.y.double()
        self.state_hash=state.state_hash; self.rtol=rank_rtol; self.rank=state.M
        self.Q,self.R=torch.linalg.qr(self.A.T,mode='reduced')
        singular=torch.linalg.svdvals(self.R)
        self.condition=float(singular.max()/singular.min())
        if float(singular.min())<=rank_rtol*float(singular.max()):
            raise ValueError(f'full physical pseudoinverse rank deficient: {self.condition}')
        self.model_target=2*self.y-self.A.sum(1)

    def project_model(self, z: torch.Tensor) -> torch.Tensor:
        """Exact affine projection [L,d], model coordinates z=2x-1; reused for O and F."""
        z=z.double()
        residual=self.model_target-z@self.A.T
        coefficients=torch.linalg.solve_triangular(self.R.T,residual.T,upper=False)
        return z+coefficients.T@self.Q.T

    def residual_rows(self, rows: torch.Tensor) -> torch.Tensor:
        rows=rows.double()
        return rows-(rows@self.Q)@self.Q.T


class PseudoinverseDecoder:
    """Centered full-row-rank pseudoinverse using float64 QR, no ridge or damping.

    Near-rank deficiency raises explicitly, never silently switches objectives.
    Residual-space update is exact for full-rank augmented [A;B]; all B derivatives retained.
    """
    def __init__(self, state: MeasurementState, mu: torch.Tensor, rank_rtol: float=1e-10, factor: MeasurementFactor | None=None):
        self.full_dim=state.A.shape[1]
        self.A=state.A.double();self.y=state.y.double();self.mu=mu.double();self.lam=0.
        self.state_hash=state.state_hash;self.rank_rtol=rank_rtol
        if factor is None:
            self.Q,self.R=torch.linalg.qr(self.A.T,mode='reduced')
            singular=torch.linalg.svdvals(self.R)
            self.condition=float(singular.max()/singular.min())
            if float(singular.min())<=rank_rtol*float(singular.max()):
                raise ValueError(f'full physical pseudoinverse rank deficient at rtol={rank_rtol}, condition={self.condition}')
        else:
            if factor.state_hash != self.state_hash or factor.rtol != rank_rtol:
                raise ValueError('stale or incompatible measurement factor')
            self.Q,self.R=factor.Q,factor.R; self.condition=factor.condition
        coeff=torch.linalg.solve_triangular(self.R.T,(self.y-self.A@self.mu)[:,None],upper=False)
        self.xhat=self.mu+(self.Q@coeff)[:,0]

    def residual_rows(self,B: torch.Tensor) -> torch.Tensor:
        B=B.double();return B-(B@self.Q)@self.Q.T

    def update_map(self,B: torch.Tensor) -> torch.Tensor:
        residual=self.residual_rows(B)
        gram=residual@residual.T
        return torch.linalg.solve(gram,residual).T

    def captured_energy_ratio(self, truth: torch.Tensor) -> dict:
        """Offline GT-only energy captured by the physical row space.

        For centered noiseless PInv, ``xhat - mu = P_A (x - mu)``, where
        ``P_A = Q Q^T`` and ``A.T = Q R`` is this decoder's thin QR.  This
        evaluates the requested ratio without materialising the d-by-d
        projector.  It must never be called by a selection policy.
        """
        if truth.numel() != self.full_dim:
            raise ValueError(f'truth has {truth.numel()} values; expected {self.full_dim}')
        centered = truth.detach().double().flatten() - self.mu.flatten()
        total = float(centered.square().sum())
        if total <= 0.:
            # A constant image equal to mu has a mathematically undefined
            # ratio, rather than an arbitrary convention such as zero.
            return dict(eta=None, captured_centered_energy=0., total_centered_energy=total,
                        M=int(self.A.shape[0]), rank=int(self.Q.shape[1]), full_dim=int(self.full_dim),
                        definition='eta = ||P_A(x-mu)||^2 / ||x-mu||^2 (undefined when denominator is zero)')
        captured = float((self.Q.T @ centered).square().sum())
        eta = captured / total
        tolerance = 1e-10
        if eta < -tolerance or eta > 1.+tolerance:
            raise RuntimeError(f'row-space energy ratio outside [0,1]: {eta}')
        eta = min(1., max(0., eta))
        return dict(eta=eta, captured_centered_energy=captured, total_centered_energy=total,
                    M=int(self.A.shape[0]), rank=int(self.Q.shape[1]), full_dim=int(self.full_dim),
                    definition='eta = ||P_A(x-mu)||^2 / ||x-mu||^2',
                    projector='P_A = Q Q^T; A^T = Q R is thin QR of actual uint8-complement effective rows',
                    domain='full physical measurement canvas; x and mu are physical intensities in [0,1]',
                    association='centered full pseudoinverse; offline GT-only evaluation, never supplied to selection')


class LinearMMSEDecoder:
    """Noiseless Wiener/LMMSE final decoder: xhat = mu + C A^T (A C A^T)^-1 (y - A mu).

    C is the circulant (stationary) prior covariance whose eigenvalues are the train-residual power
    spectrum s. Computed in whitened form B = A C^{1/2}: xhat = mu + C^{1/2} B^+ (y - A mu), via a
    float64 thin QR of B^T (same numerics and rank rule as PseudoinverseDecoder; forming A C A^T
    would square the condition number). Linear, closed form, no network, no damping; measured
    data are reproduced exactly. Near rank deficiency raises instead of being regularized.
    """
    def __init__(self, state: MeasurementState, mu: torch.Tensor, spectrum: torch.Tensor, rank_rtol: float=1e-10):
        A=state.A.double(); side=spectrum.shape[-1]; root=spectrum.to(A.device).double().sqrt()
        half=lambda X: torch.fft.ifft2(torch.fft.fft2(X.reshape(-1,side,side))*root).real.flatten(1)  # rows -> rows C^{1/2}
        Q,R=torch.linalg.qr(half(A).T,mode='reduced'); sv=torch.linalg.svdvals(R)
        self.condition=float(sv.max()/sv.min())
        if float(sv.min())<=rank_rtol*float(sv.max()):
            raise ValueError(f'LMMSE whitened rows rank deficient at rtol={rank_rtol}, condition={self.condition}')
        coeff=torch.linalg.solve_triangular(R.T,(state.y.double()-A@mu.double())[:,None],upper=False)
        self.xhat=mu.double()+half((Q@coeff).T)[0]
