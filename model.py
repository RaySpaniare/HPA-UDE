"""
HPA-UDE main model.

Includes: GeoHyperNet (static hyper-network), _ODEFunc (ODE RHS),
and HPA_UDE_Model (end-to-end model).
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Import all core components from model_components.
from model_components import (
    FiLMFluxNN,
    KANBlock,
    PhysicsHead,
    ReconstructionHead,
    SpatialKANEncoder,
    TemporalKANEncoder,
    euler_integrate_jit,
)

# Try loading torchdiffeq for adjoint ODE solving.
try:
    from torchdiffeq import odeint_adjoint as odeint  # type: ignore[import-not-found]

    _HAS_TORCHDIFFEQ = True
except Exception:
    _HAS_TORCHDIFFEQ = False


class GeoHyperNet(nn.Module):
    """
    Four-branch static encoder with terrain modulation:
    soil + terrain(gating) + location + cluster conditioning.

    Input layout: [Soil(5), Terrain(2), LonLat(2), Cluster(4)] = 13 dims.

    Physics-prior injection strategy (manual logit-bias initialization):
    Based on K-means hydro-functional zones, inject logit-space biases:
    - C1 (transition):       K_sat=0.0,  S_max=0.0,  c_exp=0.0
    - C2 (dry-hot/clayey):   K_sat=-1.0, S_max=+1.5, c_exp=+0.5
    - C3 (transition):       K_sat=0.0,  S_max=0.0,  c_exp=0.0
    - C4 (humid/sandy):      K_sat=+1.0, S_max=-1.0, c_exp=-0.5
    """

    # ===== Physics-prior bias table (from K-means analysis) =====
    # Important: K_sat and c_exp use softplus and are sensitive to logit bias.
    #   softplus(+1.0)*50 ≈ 65 mm/day (reasonable for sandy soils)
    #   softplus(-1.0)*50 ≈ 16 mm/day (reasonable for clayey soils)
    #   softplus(+0.5)*5+3 ≈ 7.6      (sandy c_exp)
    #   softplus(-0.5)*5+3 ≈ 5.2      (clayey c_exp)
    # S_max still uses sigmoid.
    # Order: [K_sat, S_max, c_exp]
    # Cluster IDs are zero-based: C1=0, C2=1, C3=2, C4=3.
    CLUSTER_PHYSICS_BIAS = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # C1: transition - neutral
            [-1.0, 1.5, 0.5],  # C2: dry-hot/clayey - low K, high S, slow drainage
            [0.0, 0.0, 0.0],  # C3: transition - neutral
            [1.0, -1.0, -0.5],  # C4: humid/sandy - high K, low S, fast drainage
        ],
        dtype=torch.float32,
    )

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        soil_dim: int = 5,  # Clay, Sand, BD, OC, Porosity
        terrain_dim: int = 2,  # Dem, Slope
        lonlat_dim: int = 2,  # Lon, Lat
        cluster_dim: int = 4,  # K-means one-hot (K=4)
    ) -> None:
        super().__init__()

        # Register physics bias buffer (no gradient updates).
        self.register_buffer("cluster_physics_bias", self.CLUSTER_PHYSICS_BIAS.clone())

        # Explicit slices for robust indexing.
        self.soil_slice = slice(0, soil_dim)
        self.terrain_slice = slice(soil_dim, soil_dim + terrain_dim)
        self.lonlat_slice = slice(soil_dim + terrain_dim, soil_dim + terrain_dim + lonlat_dim)
        self.cluster_slice = slice(
            soil_dim + terrain_dim + lonlat_dim,
            soil_dim + terrain_dim + lonlat_dim + cluster_dim,
        )

        # Save dimensions
        self.soil_dim = soil_dim
        self.terrain_dim = terrain_dim
        self.lonlat_dim = lonlat_dim
        self.cluster_dim = cluster_dim
        terrain_hidden = 32
        spatial_out_dim = 64

        # ===== Branch 1: learnable spatial encoder (Lon, Lat) =====
        self.spatial_kan = SpatialKANEncoder(
            in_dim=lonlat_dim, hidden_dim=32, out_dim=spatial_out_dim
        )

        # ===== Branch 2: soil properties =====
        self.soil_kan = KANBlock(soil_dim, hidden_dim)

        # ===== Branch 3: terrain + gating =====
        self.terrain_kan = KANBlock(terrain_dim, terrain_hidden)
        # Terrain controls which soil dimensions are emphasized.
        self.terrain_gate = nn.Sequential(
            nn.Linear(terrain_hidden, hidden_dim),
            nn.Sigmoid(),
        )

        # ===== Branch 4: cluster conditioning =====
        self.cluster_embed = nn.Linear(cluster_dim, hidden_dim, bias=False)

        # Learnable modulation on top of static physics priors.
        self.cluster_theta_mod = nn.Linear(cluster_dim, 3, bias=False)
        nn.init.zeros_(self.cluster_theta_mod.weight)

        # Deep KAN after branch fusion:
        # modulated_soil(hidden) + terrain(32) + spatial(64) + cluster(hidden)
        self.block1 = KANBlock(hidden_dim + terrain_hidden + spatial_out_dim + hidden_dim, hidden_dim)
        self.block2 = KANBlock(hidden_dim, hidden_dim)
        self.block3 = KANBlock(hidden_dim, hidden_dim)

        self.theta_head = nn.Linear(hidden_dim, 3)
        self.film_head = nn.Linear(hidden_dim, hidden_dim * 2)

        # FiLM initialization: gamma=1, beta=0.
        nn.init.zeros_(self.film_head.weight)
        nn.init.constant_(self.film_head.bias, 0.0)
        with torch.no_grad():
            self.film_head.bias[:hidden_dim].fill_(1.0)

    def forward(self, x_stat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with explicit physics-prior injection.

        Strategy:
        1. Decode cluster ID from one-hot input.
        2. Lookup cluster-specific bias [K_sat_bias, S_max_bias, c_exp_bias].
        3. Add bias to learned theta logits.
        4. Add a learnable micro-adjustment via cluster_theta_mod.
        """
        # Explicit branch split
        soil = x_stat[:, self.soil_slice]  # (B, 5)
        terrain = x_stat[:, self.terrain_slice]  # (B, 2)
        lon_lat = x_stat[:, self.lonlat_slice]  # (B, 2)
        cluster = x_stat[:, self.cluster_slice]  # (B, 4), one-hot

        # Branch feature extraction
        pos_feat = self.spatial_kan(lon_lat)  # (B, 64)
        soil_feat = self.soil_kan(soil)  # (B, hidden_dim)
        terrain_feat = self.terrain_kan(terrain)  # (B, 32)
        cluster_feat = self.cluster_embed(cluster)  # (B, hidden_dim)

        # Terrain gating on soil features
        gate = self.terrain_gate(terrain_feat)  # (B, hidden_dim)
        modulated_soil = soil_feat * gate

        # Fuse branches -> deep KAN
        h = torch.cat([modulated_soil, terrain_feat, pos_feat, cluster_feat], dim=-1)
        h = self.block3(self.block2(self.block1(h)))

        # Base physics logits learned from data
        theta_raw = self.theta_head(h)  # (B, 3)

        # Physics-prior injection from cluster lookup table
        physics_bias = torch.matmul(cluster, self.cluster_physics_bias)  # (B, 3)

        # Learnable fine-tuning (initialized to zeros)
        learned_mod = self.cluster_theta_mod(cluster)  # (B, 3)

        # Final logit = learned + prior + learned adjustment
        theta_logits = theta_raw + physics_bias + learned_mod

        # ================================================================
        # Physical-unit mapping + safety clamps:
        # logits -> absolute units (mm / mm·day^-1)
        # ================================================================

        # 1) K_sat (mm/day): positive via softplus, then clamp for safety
        k_sat_raw = F.softplus(theta_logits[:, 0:1]) * 50.0
        k_sat = torch.clamp(k_sat_raw, min=0.01, max=500.0)

        # 2) S_max (mm): constrained to (10, 90)
        s_max = torch.sigmoid(theta_logits[:, 1:2]) * 80.0 + 10.0

        # 3) c_exp (dimensionless): broad safety range [1, 30]
        c_exp_raw = F.softplus(theta_logits[:, 2:3]) * 5.0 + 3.0
        c_exp = torch.clamp(c_exp_raw, min=1.0, max=30.0)

        theta_phy = torch.cat([k_sat, s_max, c_exp], dim=-1)

        # FiLM parameters
        film = self.film_head(h)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        return theta_phy, gamma, beta


class _ODEFunc(nn.Module):
    """ODE right-hand side: dS/dt = P - E_act - D."""

    def __init__(
        self,
        p: torch.Tensor,
        pet: torch.Tensor,
        alpha: torch.Tensor,
        d_nn: torch.Tensor,
        k_sat: torch.Tensor,
        s_max: torch.Tensor,
        c_exp: torch.Tensor,
    ) -> None:
        super().__init__()
        self.p = p
        self.pet = pet
        self.alpha = alpha
        self.d_nn = d_nn
        self.k_sat = k_sat
        self.s_max = s_max
        self.c_exp = c_exp
        self.steps = p.size(1)

    def _interp(self, seq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Linear interpolation for continuous-time driver/flux estimation.
        t_clamped = torch.clamp(t, 0, self.steps - 1)
        t0 = torch.floor(t_clamped).long()
        t1 = torch.clamp(t0 + 1, max=self.steps - 1)
        w = (t_clamped - t0.float()).unsqueeze(-1)
        v0 = seq[:, t0]
        v1 = seq[:, t1]
        return v0 * (1.0 - w) + v1 * w

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        p = self._interp(self.p, t)
        pet = self._interp(self.pet, t)
        alpha = self._interp(self.alpha, t)
        d_nn = self._interp(self.d_nn, t)

        # Actual ET = PET * efficiency
        e_act = pet * alpha
        # Drainage = K_sat * (S/Smax)^c + neural correction
        d_term = self.k_sat * torch.pow(torch.clamp(s / self.s_max, 0.0, 1.2), self.c_exp) + d_nn
        return p - e_act - d_term


class HPA_UDE_Model(nn.Module):
    """HPA-UDE: static hyper-network + temporal KAN + FiLM backbone + physical ODE."""

    def __init__(
        self,
        static_dim: int = 13,
        dynamic_dim: int = 9,  # 6 meteo + 3 temporal raw [DOY_sin, DOY_cos, Year_norm]
        hidden_dim: int = 64,
        meteo_dim: int = 6,  # normalized meteorological feature dimensions
        temporal_raw_dim: int = 3,  # raw temporal feature dimensions
        temporal_embed_dim: int = 4,  # temporal KAN embedding output dimensions
        dyn_mean: Optional[torch.Tensor] = None,
        dyn_scale: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hypernet = GeoHyperNet(static_dim, hidden_dim)
        self.temporal_encoder = TemporalKANEncoder(
            in_dim=temporal_raw_dim,
            hidden_dim=16,
            out_dim=temporal_embed_dim,
        )
        backbone_in_dim = meteo_dim + temporal_embed_dim
        self.backbone = FiLMFluxNN(backbone_in_dim, hidden_dim)
        self.pretrain_head = ReconstructionHead(hidden_dim, meteo_dim)  # reconstruct meteo only
        self.physics_head = PhysicsHead(hidden_dim)
        self.dynamic_dim = dynamic_dim
        self.meteo_dim = meteo_dim
        self.temporal_raw_dim = temporal_raw_dim

        # Residual compensation branch: start near zero so ODE dominates early.
        self.res_weight = nn.Parameter(torch.tensor(-3.0))
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Dynamic normalization statistics for de-normalizing to physical units.
        if dyn_mean is None:
            dyn_mean = torch.zeros(dynamic_dim)
        if dyn_scale is None:
            dyn_scale = torch.ones(dynamic_dim)
        self.register_buffer("dyn_mean", torch.as_tensor(dyn_mean).float())
        self.register_buffer("dyn_scale", torch.as_tensor(dyn_scale).float())

    def set_dyn_stats(self, dyn_mean: torch.Tensor, dyn_scale: torch.Tensor) -> None:
        # Allow updating stats without rebuilding model.
        dyn_mean_t = torch.as_tensor(dyn_mean).float()
        dyn_scale_t = torch.as_tensor(dyn_scale).float()
        if dyn_mean_t.shape[0] != self.dynamic_dim:
            # Tolerant shape handling: pad/truncate to expected dynamic_dim.
            new_mean = torch.zeros(self.dynamic_dim)
            new_scale = torch.ones(self.dynamic_dim)
            min_dim = min(dyn_mean_t.shape[0], self.dynamic_dim)
            new_mean[:min_dim] = dyn_mean_t[:min_dim]
            new_scale[:min_dim] = dyn_scale_t[:min_dim]
            dyn_mean_t = new_mean
            dyn_scale_t = new_scale
        self.dyn_mean.copy_(dyn_mean_t)
        self.dyn_scale.copy_(dyn_scale_t)

    def _mask_inputs(self, x_dyn: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
        # Self-supervised masking to simulate missing drivers.
        if mask_ratio <= 0:
            return x_dyn
        mask = torch.rand_like(x_dyn[..., :1]) < mask_ratio
        x_dyn_masked = x_dyn.clone()
        x_dyn_masked[mask.expand_as(x_dyn_masked)] = 0.0
        return x_dyn_masked

    def _integrate_euler(
        self,
        p: torch.Tensor,
        pet: torch.Tensor,
        alpha: torch.Tensor,
        d_nn: torch.Tensor,
        k_sat: torch.Tensor,
        s_max: torch.Tensor,
        s0: torch.Tensor,
        c_exp: torch.Tensor,
    ) -> torch.Tensor:
        """Explicit Euler integration using JIT implementation."""
        k_sat_1d = k_sat.squeeze(-1)
        s_max_1d = s_max.squeeze(-1)
        c_exp_1d = c_exp.squeeze(-1)
        return euler_integrate_jit(p, pet, alpha, d_nn, k_sat_1d, s_max_1d, s0, c_exp_1d)

    def _build_backbone_input(self, x_dyn: torch.Tensor) -> torch.Tensor:
        """Split dynamic inputs into meteo + temporal and encode temporal via KAN."""
        x_meteo = x_dyn[..., : self.meteo_dim]  # (B, T, 6)
        x_temporal_raw = x_dyn[..., self.meteo_dim :]  # (B, T, 3)
        x_temporal_embed = self.temporal_encoder(x_temporal_raw)  # (B, T, 4)
        return torch.cat([x_meteo, x_temporal_embed], dim=-1)

    def forward(
        self,
        x_stat: torch.Tensor,
        x_dyn: torch.Tensor,
        mode: str = "finetune",
        adjoint: bool = False,
        y_initial: Optional[torch.Tensor] = None,
        return_flux: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass in consistent mm / mm·day^-1 units.

        Unit conventions:
          p, pet            mm/day  (de-normalized Pre_lag1/PET_lag1)
          k_sat             mm/day  softplus*50
          s_max             mm      [10, 90]
          c_exp             unitless softplus*5+3
          s_seq / pred_phy  mm      ODE integration result
          pred_sm           mm      pred_phy + residual correction
          e_act, d_term     mm/day  (flux_terms)

        Args:
            return_flux: return flux_terms for physics losses and diagnostics.

        Returns:
            - pretrain: recon (B, T, meteo_dim)
            - finetune & return_flux=True: (pred_sm, pred_phy, theta_phy, flux_terms)
            - finetune & return_flux=False: (pred_sm, pred_phy, theta_phy)
        """
        # 1) Static hyper-network: physical params + FiLM params
        theta_phy, gamma, beta = self.hypernet(x_stat)
        k_sat = theta_phy[:, 0:1]
        s_max = theta_phy[:, 1:2]
        c_exp = theta_phy[:, 2:3]

        # 2) Pretraining: masked reconstruction of dynamic drivers
        if mode == "pretrain":
            x_meteo = x_dyn[..., : self.meteo_dim]
            x_meteo_masked = self._mask_inputs(x_meteo, mask_ratio=0.5)
            # Temporal signals are not masked.
            x_temporal_raw = x_dyn[..., self.meteo_dim :]
            x_temporal_embed = self.temporal_encoder(x_temporal_raw)
            x_backbone_in = torch.cat([x_meteo_masked, x_temporal_embed], dim=-1)
            h = self.backbone(x_backbone_in, gamma, beta)
            recon = self.pretrain_head(h)
            return recon

        # 3) Finetune: build backbone input with temporal KAN encoder
        x_backbone_in = self._build_backbone_input(x_dyn)
        h = self.backbone(x_backbone_in, gamma, beta)
        alpha_et, d_nn = self.physics_head(h)

        # Use lagged precipitation/PET (t-1 affects SM at t)
        # Input order:
        # [Pre, PET, LST, LAI, Pre_lag1, PET_lag1, DOY_sin, DOY_cos, Year_norm]
        p_lag_norm = x_dyn[..., 4]
        pet_lag_norm = x_dyn[..., 5]

        # De-normalize to physical units (mm/day)
        p = p_lag_norm * self.dyn_scale[4] + self.dyn_mean[4]
        pet = pet_lag_norm * self.dyn_scale[5] + self.dyn_mean[5]

        # Hard physical non-negativity for precipitation and PET
        p = F.relu(p)
        pet = F.relu(pet)

        # Initial storage: use 0.5*Smax when unknown
        if y_initial is None:
            s0 = 0.5 * s_max.squeeze(-1)
        else:
            s0 = y_initial.squeeze(-1)

        # 4) ODE solving: prefer adjoint when requested
        if adjoint:
            if not _HAS_TORCHDIFFEQ:
                raise RuntimeError("torchdiffeq is required for adjoint integration.")
            t = torch.arange(x_dyn.size(1), device=x_dyn.device, dtype=x_dyn.dtype)
            func = _ODEFunc(
                p,
                pet,
                alpha_et.squeeze(-1),
                d_nn.squeeze(-1),
                k_sat.squeeze(-1),
                s_max.squeeze(-1),
                c_exp.squeeze(-1),
            )
            s_seq = odeint(func, s0, t, method="euler")
            s_seq = s_seq.transpose(0, 1)
            s_seq = F.relu(s_seq - 0.01) + 0.01
            s_seq = torch.minimum(s_seq, s_max * 1.2)
        else:
            s_seq = self._integrate_euler(
                p,
                pet,
                alpha_et.squeeze(-1),
                d_nn.squeeze(-1),
                k_sat,
                s_max,
                s0,
                c_exp,
            )

        # 5) Residual compensation: physics baseline + data-driven correction
        pred_phy = s_seq.unsqueeze(-1)  # (B, T, 1), mm
        res_correction = self.residual_head(h)  # (B, T, 1), mm
        pred_sm = pred_phy + torch.sigmoid(self.res_weight) * res_correction

        # 6) Optional mechanism analysis outputs
        if return_flux:
            e_act = pet.unsqueeze(-1) * alpha_et  # mm/day
            d_term = k_sat.unsqueeze(1) * torch.pow(
                torch.clamp(s_seq.unsqueeze(-1) / s_max.unsqueeze(1), 0.0, 1.2),
                c_exp.unsqueeze(1),
            ) + d_nn
            flux_terms = torch.cat([e_act, d_term], dim=-1)
            return pred_sm, pred_phy, theta_phy, flux_terms

        return pred_sm, pred_phy, theta_phy"""
HPA-UDE master model
Contains: GeoHyperNet (static hypernetwork)、_ODEFunc (ODEright-hand term)、HPA_UDE_Model (main model)
"""
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# Import all base classes from component module
from model_components import (
    KANBlock,
    SpatialKANEncoder,
    TemporalKANEncoder,
    FiLMFluxNN,
    ReconstructionHead,
    PhysicsHead,
    euler_integrate_jit,
)

# Try loading torchdiffeq for adjoint sensitivity ODE solution
try:
    from torchdiffeq import odeint_adjoint as odeint  # type: ignore[import-not-found]
    _HAS_TORCHDIFFEQ = True
except Exception:
    _HAS_TORCHDIFFEQ = False


class GeoHyperNet(nn.Module):
    """
    Four-branch terrain modulated static encoder: soil + terrain (gating) + location + clustering condition
    
    Input layout: [Soil(5), Terrain(2), LonLat(2), Cluster(4)] = 13dimension
    
    Physical prior injection strategy（Manual bias initialization）:
    Hydrologic functional partitioning based on K-means clustering, logit bias injected into parameter header（indirect physical quantity）：
    - C1 (Transition zone): K_sat=0.0,  S_max=0.0,  c_exp=0.0   (neutral, data-driven)
    - C2 (Dry heat/clay): K_sat=-1.0, S_max=+1.5, c_exp=+0.5  (Clay has strong water resistance)
    - C3 (Transition zone): K_sat=0.0,  S_max=0.0,  c_exp=0.0   (neutral)
    - C4 (moist/sandy soil): K_sat=+1.0, S_max=-1.0, c_exp=-0.5  (Sand and soil leak quickly)
    """
    # ===== Physical a priori bias table（Based on K-means analysis results）=====
    # Important: K_sat and c_exp use softplus activation, which is more sensitive to bias
    #   softplus(+1.0)*50 ≈ 65 mm/day (Reasonable value for sandy soil)
    #   softplus(-1.0)*50 ≈ 16 mm/day (Reasonable value for clay)
    #   softplus(+0.5)*5+3 ≈ 7.6      (sandy soil c_exp)
    #   softplus(-0.5)*5+3 ≈ 5.2      (clay c_exp)
    # S_max Still using sigmoid, the offset amplitude remains unchanged
    # Order: [K_sat, S_max, c_exp]
    # Note: Cluster ID starts from 0 (C1=0, C2=1, C3=2, C4=3)
    CLUSTER_PHYSICS_BIAS = torch.tensor([
        [0.0, 0.0, 0.0],     # C1: Transition Zone - Stay Neutral
        [-1.0, 1.5, 0.5],    # C2: Dry heat/clay - low hydraulic conductivity, high water storage, slow drainage
        [0.0, 0.0, 0.0],     # C3: Transition Zone - Stay Neutral
        [1.0, -1.0, -0.5],   # C4: Moist/sandy soil - high hydraulic conductivity, low water storage, quick drainage
    ], dtype=torch.float32)
    
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int,
        soil_dim: int = 5,      # Clay, Sand, BD, OC, Porosity
        terrain_dim: int = 2,   # Dem, Slope
        lonlat_dim: int = 2,    # Lon, Lat
        cluster_dim: int = 4,   # K-means One-Hot (K=4)
    ) -> None:
        super().__init__()
        
        # ===== Register physical offset as buffer（Does not participate in gradient updates）=====
        self.register_buffer("cluster_physics_bias", self.CLUSTER_PHYSICS_BIAS.clone())
        
        # ===== Explicit dimension slice definition（Avoid relative indexing problems）=====
        self.soil_slice = slice(0, soil_dim)
        self.terrain_slice = slice(soil_dim, soil_dim + terrain_dim)
        self.lonlat_slice = slice(soil_dim + terrain_dim, soil_dim + terrain_dim + lonlat_dim)
        self.cluster_slice = slice(soil_dim + terrain_dim + lonlat_dim, 
                                   soil_dim + terrain_dim + lonlat_dim + cluster_dim)
        
        # Save dimension information
        self.soil_dim = soil_dim
        self.terrain_dim = terrain_dim
        self.lonlat_dim = lonlat_dim
        self.cluster_dim = cluster_dim
        terrain_hidden = 32
        spatial_out_dim = 64  # SpatialKANEncoder Output dimensions

        # ===== Branch 1: Learnable spatial encoding (Lon, Lat) =====
        # Instead of fixed frequency SinusoidalPositionalEncoding, use KAN to adaptively learn spatial mapping
        self.spatial_kan = SpatialKANEncoder(
            in_dim=lonlat_dim, hidden_dim=32, out_dim=spatial_out_dim
        )

        # ===== Branch 2: Soil Properties =====
        self.soil_kan = KANBlock(soil_dim, hidden_dim)

        # ===== Branch 3: Terrain Properties + Gated Modulation =====
        self.terrain_kan = KANBlock(terrain_dim, terrain_hidden)
        # Gating: Terrain Features -> Soil dimension gate signal [0, 1]
        # Physical Meaning: Elevation/Slope Determination"Which dimensions of soil properties are more important?"
        self.terrain_gate = nn.Sequential(
            nn.Linear(terrain_hidden, hidden_dim),
            nn.Sigmoid(),
        )

        # ===== Branch 4: Cluster Conditioning =====
        # One-Hot (B, 4) -> Implicit parameter selection (B, hidden_dim)
        # Physical meaning: Different hydrological functional zones use different parameter offsets
        self.cluster_embed = nn.Linear(cluster_dim, hidden_dim, bias=False)
        
        # Learnable bias modulation layer（Initialized to zero, let the physical prior dominate）
        # Gradually learn fine-tuning of physical biases during training
        self.cluster_theta_mod = nn.Linear(cluster_dim, 3, bias=False)
        nn.init.zeros_(self.cluster_theta_mod.weight)  # Initially zero, fine-tuned during training

        # Deep KAN after fusion:
        # modulated_soil(hidden) + terrain(32) + spatial(64) + cluster(hidden)
        self.block1 = KANBlock(hidden_dim + terrain_hidden + spatial_out_dim + hidden_dim, hidden_dim)
        self.block2 = KANBlock(hidden_dim, hidden_dim)
        self.block3 = KANBlock(hidden_dim, hidden_dim)

        self.theta_head = nn.Linear(hidden_dim, 3)
        self.film_head = nn.Linear(hidden_dim, hidden_dim * 2)

        # FiLM Initialization: gamma=1, beta=0
        nn.init.zeros_(self.film_head.weight)
        nn.init.constant_(self.film_head.bias, 0.0)
        with torch.no_grad():
            self.film_head.bias[:hidden_dim].fill_(1.0)

    def forward(self, x_stat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        forward propagation（Contains physical prior injection）
        
        Physical a priori strategy:
        1. Decode the Cluster ID from Cluster One-Hot
        2. Look up the table to obtain the corresponding physical bias [K_sat_bias, S_max_bias, c_exp_bias]
        3. Add the bias to the theta_raw of the network output
        4. The learnable layer cluster_theta_mod provides additional fine-tuning（initially zero）
        """
        # Four-branch explicit split（Use predefined slices）
        soil = x_stat[:, self.soil_slice]         # (B, 5) Clay/Sand/BD/OC/Porosity
        terrain = x_stat[:, self.terrain_slice]   # (B, 2) Dem/Slope
        lon_lat = x_stat[:, self.lonlat_slice]    # (B, 2) Lon/Lat
        cluster = x_stat[:, self.cluster_slice]   # (B, 4) One-Hot clustering

        # Feature extraction of each branch
        pos_feat = self.spatial_kan(lon_lat)                         # (B, 64)
        soil_feat = self.soil_kan(soil)                              # (B, hidden_dim)
        terrain_feat = self.terrain_kan(terrain)                # (B, 32)
        cluster_feat = self.cluster_embed(cluster)              # (B, hidden_dim)

        # Terrain gating modulation: terrain determines which dimensions in soil are more important
        gate = self.terrain_gate(terrain_feat)                  # (B, hidden_dim)
        modulated_soil = soil_feat * gate                       # (B, hidden_dim)

        # Four fusion -> Deep KAN
        h = torch.cat([modulated_soil, terrain_feat, pos_feat, cluster_feat], dim=-1)
        h = self.block3(self.block2(self.block1(h)))

        # Physical parameters（Basic value, learned by the network）
        theta_raw = self.theta_head(h)  # (B, 3)
        
        # ===== Physical Prior Injection: Obtaining Explicit Bias from Clustering One-Hot Lookup Table =====
        # cluster: (B, 4) One-Hot -> Matrix multiplication to obtain the corresponding offset
        # cluster_physics_bias: (4, 3) -> (B, 3)
        physics_bias = torch.matmul(cluster, self.cluster_physics_bias)  # (B, 3)
        
        # Learnable fine-tuning（It is initially zero and will gradually take effect during training.）
        learned_mod = self.cluster_theta_mod(cluster)  # (B, 3)
        
        # final logit = Network output + physical prior bias + learnable fine-tuning
        theta_logits = theta_raw + physics_bias + learned_mod
        
        # =================================================================
        # Physical dimension mapping layer + safety net: Logit → absolute physical dimension (mm / mm·day⁻¹)
        # =================================================================
        
        # 1. K_sat (Saturated hydraulic conductivity mm/day):
        #    Softplus Guaranteed positive value and 50 times magnification; clamp two-way safety net to prevent gradient explosion
        k_sat_raw = F.softplus(theta_logits[:, 0:1]) * 50.0
        k_sat = torch.clamp(k_sat_raw, min=0.01, max=500.0)
        
        # 2. S_max (Maximum water storage capacity mm): 0-10cm soil typical value 40~60 mm
        #    Sigmoid strictly limited to (10, 90) mm（Corresponds to 10%~90% physical porosity）
        s_max = torch.sigmoid(theta_logits[:, 1:2]) * 80.0 + 10.0
        
        # 3. c_exp (Brooks-Corey Nonlinear index): Typical physical range 3 ~ 15
        #    Broad safety net [1, 30], compatible with clay’s extreme water retention
        c_exp_raw = F.softplus(theta_logits[:, 2:3]) * 5.0 + 3.0
        c_exp = torch.clamp(c_exp_raw, min=1.0, max=30.0)
        
        # Assembling physically constrained parameters
        theta_phy = torch.cat([k_sat, s_max, c_exp], dim=-1)

        # FiLM parameter
        film = self.film_head(h)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        return theta_phy, gamma, beta


class _ODEFunc(nn.Module):
    """ODE Right-hand term: dS/dt = P - E_act - D"""
    def __init__(
        self,
        p: torch.Tensor,
        pet: torch.Tensor,
        alpha: torch.Tensor,
        d_nn: torch.Tensor,
        k_sat: torch.Tensor,
        s_max: torch.Tensor,
        c_exp: torch.Tensor,
    ) -> None:
        super().__init__()
        self.p = p
        self.pet = pet
        self.alpha = alpha
        self.d_nn = d_nn
        self.k_sat = k_sat
        self.s_max = s_max
        self.c_exp = c_exp
        self.steps = p.size(1)

    def _interp(self, seq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Linear interpolation to ensure continuous-time drive/flux estimation
        t_clamped = torch.clamp(t, 0, self.steps - 1)
        t0 = torch.floor(t_clamped).long()
        t1 = torch.clamp(t0 + 1, max=self.steps - 1)
        w = (t_clamped - t0.float()).unsqueeze(-1)
        v0 = seq[:, t0]
        v1 = seq[:, t1]
        return v0 * (1.0 - w) + v1 * w

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        p = self._interp(self.p, t)
        pet = self._interp(self.pet, t)
        alpha = self._interp(self.alpha, t)
        d_nn = self._interp(self.d_nn, t)

        # Physical meaning: actual evapotranspiration = PET * evapotranspiration efficiency
        e_act = pet * alpha
        # Physical meaning: drainage term = Hydraulic conductivity * (S/Smax)^c + Neural network correction
        d_term = self.k_sat * torch.pow(torch.clamp(s / self.s_max, 0.0, 1.2), self.c_exp) + d_nn
        return p - e_act - d_term


class HPA_UDE_Model(nn.Module):
    """HPA-UDE Main model: static super network + learnable time KAN + FiLM dynamic backbone + physical ODE"""
    def __init__(
        self,
        static_dim: int = 13,
        dynamic_dim: int = 9,   # 6 meteo + 3 temporal raw [DOY_sin, DOY_cos, Year_norm]
        hidden_dim: int = 64,
        meteo_dim: int = 6,              # Standardized meteorological variable dimensions
        temporal_raw_dim: int = 3,       # Original time feature dimension
        temporal_embed_dim: int = 4,     # KAN Time embedding output dimension
        dyn_mean: Optional[torch.Tensor] = None,
        dyn_scale: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hypernet = GeoHyperNet(static_dim, hidden_dim)
        self.temporal_encoder = TemporalKANEncoder(
            in_dim=temporal_raw_dim,
            hidden_dim=16,
            out_dim=temporal_embed_dim,
        )
        backbone_in_dim = meteo_dim + temporal_embed_dim
        self.backbone = FiLMFluxNN(backbone_in_dim, hidden_dim)
        self.pretrain_head = ReconstructionHead(hidden_dim, meteo_dim)  # Reconstruct only meteorological variables
        self.physics_head = PhysicsHead(hidden_dim)
        self.dynamic_dim = dynamic_dim
        self.meteo_dim = meteo_dim
        self.temporal_raw_dim = temporal_raw_dim

        # Residual compensation branch: the initial weight is 0, and the initial stage of training is completely dominated by physical ODE
        self.res_weight = nn.Parameter(torch.tensor(-3.0))
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Standardized statistics for dynamic variables（Used to denormalize to physical dimensions）
        if dyn_mean is None:
            dyn_mean = torch.zeros(dynamic_dim)
        if dyn_scale is None:
            dyn_scale = torch.ones(dynamic_dim)
        self.register_buffer("dyn_mean", torch.as_tensor(dyn_mean).float())
        self.register_buffer("dyn_scale", torch.as_tensor(dyn_scale).float())

    def set_dyn_stats(self, dyn_mean: torch.Tensor, dyn_scale: torch.Tensor) -> None:
        # Statistics can be updated during the training phase（Avoid rebuilding models when data is leaked）
        # Perform fault-tolerant processing of dimension checks
        dyn_mean_t = torch.as_tensor(dyn_mean).float()
        dyn_scale_t = torch.as_tensor(dyn_scale).float()
        if dyn_mean_t.shape[0] != self.dynamic_dim:
            # If dimensions do not match, expand to match expected dimensions（Fill with 0s and 1s）
            new_mean = torch.zeros(self.dynamic_dim)
            new_scale = torch.ones(self.dynamic_dim)
            min_dim = min(dyn_mean_t.shape[0], self.dynamic_dim)
            new_mean[:min_dim] = dyn_mean_t[:min_dim]
            new_scale[:min_dim] = dyn_scale_t[:min_dim]
            dyn_mean_t = new_mean
            dyn_scale_t = new_scale
        self.dyn_mean.copy_(dyn_mean_t)
        self.dyn_scale.copy_(dyn_scale_t)

    def _mask_inputs(self, x_dyn: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
        # Self-supervised masks: Simulating missingness drivers
        if mask_ratio <= 0:
            return x_dyn
        mask = torch.rand_like(x_dyn[..., :1]) < mask_ratio
        x_dyn_masked = x_dyn.clone()
        x_dyn_masked[mask.expand_as(x_dyn_masked)] = 0.0
        return x_dyn_masked

    def _integrate_euler(
        self,
        p: torch.Tensor,
        pet: torch.Tensor,
        alpha: torch.Tensor,
        d_nn: torch.Tensor,
        k_sat: torch.Tensor,
        s_max: torch.Tensor,
        s0: torch.Tensor,
        c_exp: torch.Tensor,
    ) -> torch.Tensor:
        """Explicit Euler Integral: Invoke JIT compiled version to reduce Python GIL overhead"""
        k_sat_1d = k_sat.squeeze(-1)
        s_max_1d = s_max.squeeze(-1)
        c_exp_1d = c_exp.squeeze(-1)
        return euler_integrate_jit(p, pet, alpha, d_nn, k_sat_1d, s_max_1d, s0, c_exp_1d)

    def _build_backbone_input(self, x_dyn: torch.Tensor) -> torch.Tensor:
        """Split the dynamic input into weather + time, process it through the KAN time encoder and then splice it"""
        x_meteo = x_dyn[..., :self.meteo_dim]                    # (B, T, 6)
        x_temporal_raw = x_dyn[..., self.meteo_dim:]             # (B, T, 3)
        x_temporal_embed = self.temporal_encoder(x_temporal_raw) # (B, T, 4)
        return torch.cat([x_meteo, x_temporal_embed], dim=-1)    # (B, T, 10)

    def forward(
        self,
        x_stat: torch.Tensor,
        x_dyn: torch.Tensor,
        mode: str = "finetune",
        adjoint: bool = False,
        y_initial: Optional[torch.Tensor] = None,
        return_flux: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """
        forward propagation（Whole journey mm/mm·day⁻¹ dimension）

        Unit agreement:
          p, pet            mm/day  (Pre_lag1/PET_lag1 denormalization)
          k_sat             mm/day  softplus*50 (Positive value, no upper bound)
          s_max             mm      [10, 90]
          c_exp             dimensionless softplus*5+3 (≥3)
          s_seq / pred_phy  mm      (ODE points result)
          pred_sm           mm      (= pred_phy + sigmoid(res_weight) * residual)
          e_act, d_term     mm/day  (flux_terms)

        Args:
            return_flux: Whether to return flux_terms（Physical constraints loss / required for mechanism analysis）

        Returns:
            - pretrain Mode: recon (B, T, meteo_dim), normalized space
            - finetune Pattern return_flux=True:  (pred_sm, pred_phy, theta_phy, flux_terms)
            - finetune Pattern return_flux=False: (pred_sm, pred_phy, theta_phy)
              pred_sm:    (B, T, 1)  mm
              pred_phy:   (B, T, 1)  mm  (Pure ODE, no residuals)
              theta_phy:  (B, 3)     [k_sat mm/day, s_max mm, c_exp]
              flux_terms: (B, T, 2)  mm/day  [:,:,0]=e_act, [:,:,1]=d_term
        """
        # 1) Static hypernetwork: generating physical and FiLM parameters
        theta_phy, gamma, beta = self.hypernet(x_stat)
        k_sat = theta_phy[:, 0:1]
        s_max = theta_phy[:, 1:2]
        c_exp = theta_phy[:, 2:3]

        # 2) Pre-training: dynamic driver of mask reconstruction（Reconstruct meteorological variables only）
        if mode == "pretrain":
            x_meteo = x_dyn[..., :self.meteo_dim]
            x_meteo_masked = self._mask_inputs(x_meteo, mask_ratio=0.5)
            # Time encoding is not masked（always known）
            x_temporal_raw = x_dyn[..., self.meteo_dim:]
            x_temporal_embed = self.temporal_encoder(x_temporal_raw)
            x_backbone_in = torch.cat([x_meteo_masked, x_temporal_embed], dim=-1)
            h = self.backbone(x_backbone_in, gamma, beta)
            recon = self.pretrain_head(h)
            return recon

        # 3) Fine-tuning: Building backbone input via KAN temporal encoder
        x_backbone_in = self._build_backbone_input(x_dyn)
        h = self.backbone(x_backbone_in, gamma, beta)
        alpha_et, d_nn = self.physics_head(h)

        # Remove delayed rainfall and potential evapotranspiration（t-1 The value at time affects the SM at time t）
        # Dynamic input sequence: [Pre, PET, LST, LAI, Pre_lag1, PET_lag1, DOY_sin, DOY_cos, Year_norm]
        # Meteorological part (0-5): Standardized driving variables
        # Temporal part (6-8): Raw input to KAN temporal encoder（Handled internally by _build_backbone_input）
        p_lag_norm = x_dyn[..., 4]   # Pre_lag1 (t-1)
        pet_lag_norm = x_dyn[..., 5] # PET_lag1 (t-1)

        # Denormalization to physical dimensions（mm/day）
        # NOTE: The lagged variable uses the same normalization parameters as the original variable (indexes 0 and 1)
        p = p_lag_norm * self.dyn_scale[4] + self.dyn_mean[4]
        pet = pet_lag_norm * self.dyn_scale[5] + self.dyn_mean[5]

        # Physical hard constraints: rainfall and evapotranspiration are non-negative
        p = F.relu(p)
        pet = F.relu(pet)

        # Initial moisture content: if unknown, use 0.5*Smax
        if y_initial is None:
            s0 = 0.5 * s_max.squeeze(-1)
        else:
            s0 = y_initial.squeeze(-1)

        # 4) ODE Solution: Use adjoint first to reduce video memory
        if adjoint:
            if not _HAS_TORCHDIFFEQ:
                raise RuntimeError("torchdiffeq is required for adjoint integration.")
            t = torch.arange(x_dyn.size(1), device=x_dyn.device, dtype=x_dyn.dtype)
            func = _ODEFunc(
                p,
                pet,
                alpha_et.squeeze(-1),
                d_nn.squeeze(-1),
                k_sat.squeeze(-1),
                s_max.squeeze(-1),
                c_exp.squeeze(-1),
            )
            s_seq = odeint(func, s0, t, method="euler")
            s_seq = s_seq.transpose(0, 1)
            s_seq = F.relu(s_seq - 0.01) + 0.01
            s_seq = torch.minimum(s_seq, s_max * 1.2)
        else:
            s_seq = self._integrate_euler(
                p,
                pet,
                alpha_et.squeeze(-1),
                d_nn.squeeze(-1),
                k_sat,
                s_max,
                s0,
                c_exp,
            )

        # 5) Residual compensation: physical ODE guarantee + data-driven refinement（Dimension in mm throughout）
        pred_phy = s_seq.unsqueeze(-1)   # (B, T, 1)  ODE Integration result, unit: mm
        res_correction = self.residual_head(h)  # (B, T, 1)  Residual correction, unit: mm
        pred_sm = pred_phy + torch.sigmoid(self.res_weight) * res_correction  # (B, T, 1)  mm

        # 6) Mechanism analysis（Optional）：Save actual evapotranspiration and drainage fluxes（mm/day，physical dimensions）
        if return_flux:
            e_act = pet.unsqueeze(-1) * alpha_et            # mm/day
            d_term = k_sat.unsqueeze(1) * torch.pow(
                torch.clamp(s_seq.unsqueeze(-1) / s_max.unsqueeze(1), 0.0, 1.2),
                c_exp.unsqueeze(1),
            ) + d_nn                                         # mm/day
            flux_terms = torch.cat([e_act, d_term], dim=-1)
            # pred_sm: mm（final prediction）pred_phy: mm（Pure ODE results）theta_phy: [mm/day, mm, -]
            return pred_sm, pred_phy, theta_phy, flux_terms

        return pred_sm, pred_phy, theta_phy
