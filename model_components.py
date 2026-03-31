"""
Core components for HPA-UDE.

Includes: KAN layers, spatial/temporal KAN encoders, Mamba blocks,
FiLM flux network, physics head, and JIT Euler integration.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


# ==================== JIT-Optimized Euler Integration ====================
@torch.jit.script
def _euler_step(
    s: torch.Tensor,
    p_t: torch.Tensor,
    pet_t: torch.Tensor,
    alpha_t: torch.Tensor,
    d_nn_t: torch.Tensor,
    k_sat: torch.Tensor,
    s_max: torch.Tensor,
    c_exp: torch.Tensor,
) -> torch.Tensor:
    """Single Euler integration step with unit and boundary safeguards."""
    # Actual evapotranspiration = PET * efficiency
    e_act = pet_t * alpha_t
    # Clamp relative saturation to allow mild ponding and avoid instability.
    s_ratio = torch.clamp(s / s_max, 0.0, 1.2)
    d_term = k_sat * torch.pow(s_ratio, c_exp) + d_nn_t
    # Water balance in mm units
    s_new = s + p_t - e_act - d_term
    # Soft non-negativity with a small floor to avoid divide-by-zero.
    s_new = F.relu(s_new - 0.01) + 0.01
    # Hard upper bound for physically plausible storage.
    s_new = torch.minimum(s_new, s_max * 1.2)
    return s_new


@torch.jit.script
def euler_integrate_jit(
    p: torch.Tensor,
    pet: torch.Tensor,
    alpha: torch.Tensor,
    d_nn: torch.Tensor,
    k_sat: torch.Tensor,
    s_max: torch.Tensor,
    s0: torch.Tensor,
    c_exp: torch.Tensor,
) -> torch.Tensor:
    """JIT Euler integration with preallocated output (no dynamic list)."""
    bsz = p.size(0)
    steps = p.size(1)
    # Preallocate to avoid append + stack overhead.
    outputs = torch.empty(bsz, steps, device=p.device, dtype=p.dtype)
    s = s0

    for t in range(steps):
        s = _euler_step(
            s, p[:, t], pet[:, t], alpha[:, t], d_nn[:, t],
            k_sat, s_max, c_exp
        )
        outputs[:, t] = s

    return outputs


# ==================== KAN Components ====================
class ChebyKANLinear(nn.Module):
    """Chebyshev-polynomial KAN linear layer with linear residual branch."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        degree: int = 3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree
        # Linear branch: stable baseline + nonlinear refinement.
        self.weight_linear = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        # Polynomial branch weights: (out_dim, in_dim, degree+1)
        self.poly_weight = nn.Parameter(
            torch.randn(out_dim, in_dim, degree + 1) / (in_dim * (degree + 1) + 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) -> (B, out_dim)"""
        # Linear branch
        linear = F.linear(x, self.weight_linear, self.bias)
        # Normalize to [-1, 1] for Chebyshev basis domain.
        x_norm = torch.tanh(x)
        # Chebyshev basis: T_0=1, T_1=x, T_n=2x*T_{n-1}-T_{n-2}
        # Use list + stack to avoid in-place autograd issues.
        t0 = torch.ones_like(x_norm)
        t1 = x_norm
        basis_list = [t0, t1]
        for _ in range(2, self.degree + 1):
            basis_list.append(2 * x_norm * basis_list[-1] - basis_list[-2])
        basis = torch.stack(basis_list, dim=-1)
        # einsum: (B, in_dim, degree+1) x (out_dim, in_dim, degree+1) -> (B, out_dim)
        poly_out = torch.einsum("bid,oid->bo", basis, self.poly_weight)
        return linear + poly_out


class KANBlock(nn.Module):
    """Basic KAN block: KANLinear + LayerNorm + activation."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.kan = ChebyKANLinear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.kan(x)))


# ==================== Positional Encoding ====================
class SinusoidalPositionalEncoding(nn.Module):
    """[Deprecated] Maps lon/lat to periodic space; replaced by SpatialKANEncoder."""

    def __init__(self, in_dim: int = 2, out_dim: int = 32, max_freq: float = 10.0) -> None:
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError("out_dim must be even for sinusoidal encoding.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        half = out_dim // 2
        freqs = torch.linspace(1.0, max_freq, half)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2) -> (B, out_dim)
        x = x.unsqueeze(-1)
        angles = x * self.freqs
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return torch.cat([sin, cos], dim=-1).reshape(x.size(0), -1)


# ==================== Learnable Spatial Encoder ====================
class SpatialKANEncoder(nn.Module):
    """
    Hybrid spatial encoder: fixed multi-frequency sin/cos basis + KAN refinement.

    Design:
    1. sin/cos expansion provides stable periodic inductive bias.
    2. KAN layers learn nonlinear longitude-latitude interactions.
    3. Residual-style pathway keeps training stable in early epochs.

    Input: [Lon, Lat] (normalized)
    Output: (B, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 32,
        out_dim: int = 64,
        n_freqs: int = 8,
        max_freq: float = 10.0,
    ) -> None:
        super().__init__()
        # Fixed sin/cos basis: 2 * in_dim * n_freqs
        self.n_freqs = n_freqs
        freqs = torch.linspace(1.0, max_freq, n_freqs)
        self.register_buffer("freqs", freqs)
        sincos_dim = in_dim * n_freqs * 2

        # KAN refinement for nonlinear interactions on top of sin/cos basis.
        self.kan1 = KANBlock(sincos_dim + in_dim, hidden_dim)
        self.kan2 = KANBlock(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2) -> (B, out_dim)"""
        # Multi-frequency sin/cos expansion.
        x_expanded = x.unsqueeze(-1) * self.freqs  # (B, 2, n_freqs)
        sincos = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)
        sincos_flat = sincos.reshape(x.size(0), -1)
        # Concatenate raw lon/lat to preserve linear information.
        h_in = torch.cat([sincos_flat, x], dim=-1)
        return self.kan2(self.kan1(h_in))


# ==================== Learnable Temporal Encoder ====================
class TemporalKANEncoder(nn.Module):
    """
    Temporal encoder: base sin/cos features + KAN residual refinement.

    Design:
    1. Keep DOY_sin/DOY_cos as lossless annual-cycle baseline.
    2. Use KAN residual to model asymmetric seasonal effects.
    3. Keep Year_norm as pass-through to avoid overfitting sparse years.

    Input: [DOY_sin, DOY_cos, Year_norm]
    Output: [refined_sin, refined_cos, year_norm, learned_interaction]
    """

    def __init__(self, in_dim: int = 3, hidden_dim: int = 16, out_dim: int = 4) -> None:
        super().__init__()
        # Residual branch for temporal corrections and cross-feature interactions.
        self.kan_refine = KANBlock(in_dim, hidden_dim)
        self.kan_out = KANBlock(hidden_dim, out_dim)
        # Residual gate starts small and grows during training.
        self.res_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 3) = [DOY_sin, DOY_cos, Year_norm]
        Returns: (B, T, out_dim)
        """
        if x.dim() == 3:
            bsz, steps, dims = x.shape
            x_flat = x.reshape(bsz * steps, dims)
            out = self._encode(x_flat)
            return out.reshape(bsz, steps, -1)
        return self._encode(x)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Core encoding: base pass-through + gated KAN residual."""
        base = x
        kan_res = self.kan_out(self.kan_refine(x))
        gate = torch.sigmoid(self.res_gate)
        out_dim = kan_res.size(-1)
        in_dim = x.size(-1)
        if out_dim <= in_dim:
            return base[:, :out_dim] + gate * kan_res
        # out_dim > in_dim: first in_dim dimensions are residual-mixed,
        # additional dimensions are pure learned increments.
        left = gate * kan_res[:, :in_dim] + base
        right = gate * kan_res[:, in_dim:]
        return torch.cat([left, right], dim=-1)


# ==================== Mamba Components ====================
# Try mamba_ssm; fall back to MinimalMamba when unavailable.
try:
    from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore[import-not-found]

    _HAS_MAMBA = True
except Exception:
    _HAS_MAMBA = False


class MinimalMamba(nn.Module):
    """Lightweight Mamba fallback based on Conv1d + gating + residual."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.dwconv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
        )
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(0.1)

        # Initialization tuned for stability.
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.kaiming_uniform_(self.dwconv.weight, a=math.sqrt(5))
        nn.init.zeros_(self.dwconv.bias)
        # Small output projection init to let residual dominate early training.
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        u, v = self.in_proj(x).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)
        y = u * torch.sigmoid(v)
        y = self.dropout(self.out_proj(y))
        return y + residual


class MambaBlock(nn.Module):
    """Mamba block: use mamba_ssm if available, else MinimalMamba."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)  # Pre-norm
        if _HAS_MAMBA:
            self.core = Mamba(d_model=d_model)
            self.has_internal_residual = False
        else:
            self.core = MinimalMamba(d_model=d_model)
            self.has_internal_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = self.norm(x)
        h = self.core(h)
        # MinimalMamba already includes residual; mamba_ssm does not.
        if not self.has_internal_residual:
            h = h + x
        return h


# ==================== FiLM Dynamic Flux Network ====================
class FiLMFluxNN(nn.Module):
    """Dynamic flux backbone: Mamba + FiLM modulation with residual blending."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.mamba1 = MambaBlock(hidden_dim)
        self.mamba2 = MambaBlock(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

        # Learnable FiLM intensity (starts as mild modulation).
        self.film_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.7)
        nn.init.zeros_(self.in_proj.bias)

    def forward(self, x_dyn: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # We intentionally do not feed discrete SM_{t-1} directly here.
        # State dependence is already explicitly modeled in physical equation D=f(S).
        h = self.in_proj(x_dyn)
        h = self.mamba1(h)
        h = self.mamba2(h)
        h = self.norm(h)

        # Soft FiLM modulation: residual interpolation for stable control.
        film_modulation = gamma.unsqueeze(1) * h + beta.unsqueeze(1)
        h = h + self.film_scale * (film_modulation - h)
        h = self.dropout(self.act(h))
        return h


# ==================== Output Heads ====================
class ReconstructionHead(nn.Module):
    """Self-supervised reconstruction head for masked dynamic variables."""

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class PhysicsHead(nn.Module):
    """
    Physics head (enhanced): outputs ET efficiency and drainage correction.

    Improvements:
    - Adds a 2-layer shared trunk to improve flux estimation quality.
    - alpha_et in [0, 1] via sigmoid.
    - d_nn in [0, 50] mm/day via softplus + clamp.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        mid_dim = hidden_dim // 2
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.SiLU(),
            nn.LayerNorm(mid_dim),
        )
        # Two output heads
        self.alpha_head = nn.Linear(mid_dim, 1)
        self.drain_head = nn.Linear(mid_dim, 1)

        # Initialization: alpha around 0.5, d_nn near small positive values.
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.drain_head.bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(h)
        alpha_et = torch.sigmoid(self.alpha_head(feat))
        # d_nn non-negative and upper bounded for numerical safety.
        d_nn = torch.clamp(F.softplus(self.drain_head(feat)), max=50.0)
        return alpha_et, d_nn