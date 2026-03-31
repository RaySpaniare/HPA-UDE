"""
Physical Constraint Loss Function Module
=====================
Contains the physically constrained loss function required for HPA-UDE model training

Loss function:
1. Stepwise Mass Conservation Loss
2. Flux Boundary Loss
"""
import torch
import torch.nn.functional as F


def calc_mass_conservation_loss(
    pred_sm: torch.Tensor,
    p: torch.Tensor,
    e_act: torch.Tensor,
    d_term: torch.Tensor,
) -> torch.Tensor:
    """
        Stepwise mass conservation loss（mm consistent dimensions）：

        Physics equation: dS/dt = P - E - D
            Left: pred_sm unit mm, change mm/day
            Right: Flux unit mm/day

        Loss = MSE(ΔS_t, P_t - E_t - D_t)
    
    Args:
        pred_sm: Predict soil moisture content (B, T) or (B, T, 1)
        p: Precipitation mm/day (B, T)
        e_act: Actual evapotranspiration mm/day (B, T)
        d_term: Drainage flux mm/day (B, T)
    """
    if pred_sm.dim() == 3:
        pred_sm = pred_sm.squeeze(-1)
    if e_act.dim() == 3:
        e_act = e_act.squeeze(-1)
    if d_term.dim() == 3:
        d_term = d_term.squeeze(-1)
    
    # Stepwise soil moisture change dS = S_{t+1} - S_t  (Unit: mm/day)
    delta_s = pred_sm[:, 1:] - pred_sm[:, :-1]

    # flux balance（Unit: mm/day）
    flux_balance = p[:, :-1] - e_act[:, :-1] - d_term[:, :-1]
    
    return F.mse_loss(delta_s, flux_balance)


def calc_flux_boundary_loss(
    e_act: torch.Tensor,
    pet: torch.Tensor,
) -> torch.Tensor:
    """
    Flux boundary constraint loss: ensure 0 ≤ E_act ≤ PET
    Loss_flux = mean(ReLU(-E_act)) + mean(ReLU(E_act - PET))
    """
    if e_act.dim() == 3:
        e_act = e_act.squeeze(-1)
    if pet.dim() == 3:
        pet = pet.squeeze(-1)
    
    loss_lower = torch.mean(F.relu(-e_act))
    loss_upper = torch.mean(F.relu(e_act - pet))
    return loss_lower + loss_upper


def calc_weighted_huber_loss(
    pred: torch.Tensor,
    obs: torch.Tensor,
    huber_beta: float = 2.0,
    delta_alpha: float = 0.05,
) -> torch.Tensor:
    """
    Rate of Change Weighted Huber Loss（Dynamic Weighting + Gradient Focus）

    1. Huber Core: Use MSE for small errors, switch to MAE for large errors（Prevent extreme events from dominating gradients）
    2. Change rate weighting: Periods in which SM changes rapidly have higher weights（Focus on rainfall infiltration/increasing drought）

    Args:
        pred: Predicted value (B, T) or (B, T, 1) in mm
        obs: Observed value (B, T) or (B, T, 1), unit: mm
        huber_beta: Huber threshold（mm dimension）
                    2.0mm ≈ 2.5% of typical SM range (0-80mm), reasonable small/large error cutoff
        delta_alpha: rate of change weighted intensity（mm dimension）
                    0.05：Typical daily variation value: 0-10mm/day,
                    Weight range = [1, 1+0.05×10] = [1, 1.5x]，Stable and non-explosive
    """
    if pred.dim() == 3:
        pred = pred.squeeze(-1)
    if obs.dim() == 3:
        obs = obs.squeeze(-1)

    # 1. Element-wise Huber Loss
    error = pred - obs
    abs_error = torch.abs(error)
    huber = torch.where(
        abs_error < huber_beta,
        0.5 * error ** 2,
        huber_beta * (abs_error - 0.5 * huber_beta),
    )

    # 2. Rate of change weighting:|SM_{t} - SM_{t-1}| The larger the value, the higher the weight.
    delta_obs = torch.abs(obs[:, 1:] - obs[:, :-1])
    weight = torch.ones_like(obs)
    weight[:, 1:] = 1.0 + delta_alpha * delta_obs

    return (weight * huber).mean()


# ================================================================
# 【New】KAN Physical Monotonicity Constraint Loss
# ================================================================
# static feature index（Strictly corresponds to STATIC_COLS in dataset_config.py）
# Input layout: [Clay(0), Sand(1), BD(2), OC(3), Porosity(4),
#            Dem(5), Slope(6), Lon(7), Lat(8), Cluster_OH(9:13)]
_FEAT_IDX_BD = 2      # Bulk Density Test weight
_FEAT_IDX_SAND = 1    # Sand Content sand content
_PARAM_IDX_KSAT = 0   # GeoHyperNet Output: K_sat
_PARAM_IDX_SMAX = 1   # GeoHyperNet Output: S_max


def calc_monotonicity_loss(
    geo_hypernet: nn.Module,
    x_stat: torch.Tensor,
) -> torch.Tensor:
    """
    KAN Physical monotonicity constrained loss（Monotonicity Constraint Loss via Autograd）。

    Input-output mapping of GeoHyperNet (KAN) via autograd at each weight update
    The following two soil hydraulic physics prior penalties are imposed:

    L_mono = E[ ReLU(∂S_max/∂BD) ] + E[ ReLU(-∂K_sat/∂Sand) ]

    Constraint 1: ∂S_max/∂BD ≤ 0
        Physical explanation: The larger the bulk density (BD) → the tighter the soil → the porosity decreases → the smaller the maximum water storage capacity (S_max)

    Constraint 2: ∂K_sat/∂Sand ≥ 0
        Physical explanation: The higher the sand content (Sand) → the more macropores → the greater the saturated hydraulic conductivity (K_sat)

    Implementation strategy:
        1. Clone x_stat and enable requires_grad（Does not interfere with the main calculation graph）
        2. Forward propagation gets theta_phy = [K_sat, S_max, c_exp]
        3. Extract partial derivatives with torch.autograd.grad
        4. ReLU Penalizes gradient directions that violate physics

    Args:
        geo_hypernet: GeoHyperNet Example（KAN Static parameter generator）
        x_stat: (B, 13) Static input feature, float32

    Returns:
        loss_mono: Scalar tensor, can be directly involved in total_loss.backward()

    References:
        Beucler et al. (2021), PRL — "Enforcing Analytic Constraints in NNs"
    """
    # Step 1: Create independent computation graph branches（Does not affect back propagation of main loss）
    x = x_stat.detach().clone().requires_grad_(True)

    # Step 2: Forward propagation → theta_phy: (B, 3) = [K_sat, S_max, c_exp]
    theta_phy, _gamma, _beta = geo_hypernet(x)

    # Step 3: extract ∂S_max/∂BD
    # Sum batch to make it a scalar（Each sample is independent and the cross gradient is 0）
    s_max_sum = theta_phy[:, _PARAM_IDX_SMAX].sum()
    grad_smax = torch.autograd.grad(
        outputs=s_max_sum, inputs=x,
        create_graph=True, retain_graph=True,
    )[0]  # (B, 13)
    d_Smax_d_BD = grad_smax[:, _FEAT_IDX_BD]  # (B,)

    # Step 4: extract ∂K_sat/∂Sand
    k_sat_sum = theta_phy[:, _PARAM_IDX_KSAT].sum()
    grad_ksat = torch.autograd.grad(
        outputs=k_sat_sum, inputs=x,
        create_graph=True, retain_graph=True,
    )[0]  # (B, 13)
    d_Ksat_d_Sand = grad_ksat[:, _FEAT_IDX_SAND]  # (B,)

    # Step 5: ReLU Penalizes gradient directions that violate physics
    # Constraint 1: ∂S_max/∂BD ≤ 0 → punish > 0 part of
    penalty_smax = F.relu(d_Smax_d_BD).mean()
    # Constraint 2: ∂K_sat/∂Sand ≥ 0 → punish < 0 part of
    penalty_ksat = F.relu(-d_Ksat_d_Sand).mean()

    loss_mono = penalty_smax + penalty_ksat
    return loss_mono

