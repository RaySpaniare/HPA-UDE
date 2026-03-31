"""
KAN Native KAN Symbolic Regression module (Native KAN Symbolic Regression)
=====================================================
Extract interpretable formulas directly from ChebyKAN weights without PySR/Julia dependencies

Core principles:
  φ_i(x) = w_lin·x + Σ_k c_k T_k(tanh(x))
  Read {w_lin, c_k} after training → cheb2poly → sympy expression → Prune → LaTeX

use:
  A) Drought index formula discovery: lightweight SymbolicKAN alternative to PySR
  B) Model Interpretability: Extracting KAN Activation Curves + Feature Importance from HPA-UDE’s GeoHyperNet
"""
import os, warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.polynomial.chebyshev import cheb2poly
from sklearn.metrics import mean_squared_error, r2_score

try:
    import sympy as sp
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False
    warnings.warn("sympy not installed. Run: pip install sympy")


# ====================== Part 1: Chebyshev → symbolic formula ======================

def build_univariate_expr(
    linear_w: float, cheb_coeffs: np.ndarray,
    var_expr: "sp.Expr", threshold: float = 1e-4,
) -> "sp.Expr":
    """φ(x) = w_lin·x + Σ_k a_k·tanh(x)^k  (a_k via cheb2poly)"""
    mono = cheb2poly(cheb_coeffs)
    u = sp.tanh(var_expr)
    expr = sp.Float(0)
    for k, a_k in enumerate(mono):
        if abs(a_k) >= threshold:
            expr += sp.Float(round(float(a_k), 6)) * (1 if k == 0 else u**k)
    if abs(linear_w) >= threshold:
        expr += sp.Float(round(float(linear_w), 6)) * var_expr
    return expr


def extract_kan_layer_formulas(
    weight_linear: np.ndarray, bias: np.ndarray,
    poly_weight: np.ndarray, var_names: List[str],
    output_names: Optional[List[str]] = None, threshold: float = 1e-4,
) -> Dict[str, "sp.Expr"]:
    """Extract full symbolic formula from ChebyKANLinear layer weights: y_j = b_j + Σ_i φ_{ji}(x_i)"""
    out_dim, in_dim, _ = poly_weight.shape
    if output_names is None:
        output_names = [f"y_{j}" for j in range(out_dim)]
    x_syms = [sp.Symbol(n) for n in var_names]
    formulas = {}
    for j in range(out_dim):
        expr = sp.Float(round(float(bias[j]), 6)) if abs(bias[j]) >= threshold else sp.Float(0)
        for i in range(in_dim):
            expr += build_univariate_expr(
                float(weight_linear[j, i]), poly_weight[j, i, :], x_syms[i], threshold)
        formulas[output_names[j]] = expr
    return formulas


def compute_feature_importance(w_lin: np.ndarray, poly_w: np.ndarray) -> np.ndarray:
    """Importance_i = Σ_j (|w_lin_{ji}| + Σ_k |c_{jik}|)"""
    return np.sum(np.abs(w_lin), axis=0) + np.sum(np.abs(poly_w), axis=(0, 2))


def prune_expression(expr: "sp.Expr", threshold: float = 0.01) -> "sp.Expr":
    """Remove absolute value coefficients < threshold addition term"""
    if not _HAS_SYMPY:
        return expr
    kept = [t for t in sp.Add.make_args(expr) if abs(float(t.as_coeff_Mul()[0])) >= threshold]
    return sp.Add(*kept) if kept else sp.Float(0)


# ====================== Part 2: Lightweight SymbolicKAN ======================

class _ChebyKANLinear(nn.Module):
    """Standalone ChebyKAN layer"""
    def __init__(self, in_dim: int, out_dim: int, degree: int = 4) -> None:
        super().__init__()
        self.in_dim, self.out_dim, self.degree = in_dim, out_dim, degree
        self.weight_linear = nn.Parameter(torch.randn(out_dim, in_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.poly_weight = nn.Parameter(
            torch.randn(out_dim, in_dim, degree + 1) / (in_dim * (degree + 1) + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = F.linear(x, self.weight_linear, self.bias)
        u = torch.tanh(x)
        T_prev, T_curr = torch.ones_like(u), u
        basis = [T_prev, T_curr]
        for _ in range(2, self.degree + 1):
            T_next = 2 * u * T_curr - T_prev
            basis.append(T_next)
            T_prev, T_curr = T_curr, T_next
        return linear + torch.einsum("bid,oid->bo", torch.stack(basis, dim=-1), self.poly_weight)


class SymbolicKAN(nn.Module):
    """
    Formula discovery using lightweight KAN（GAM structure）: y = Σ_i φ_i(x_i) + bias
    Optional pairwise interaction features, L1 regularization encourages sparsity
    """
    def __init__(self, in_dim: int, out_dim: int = 1, degree: int = 4, interaction: bool = True):
        super().__init__()
        self.in_dim_raw, self.interaction = in_dim, interaction
        n_interact = in_dim * (in_dim - 1) // 2 if interaction else 0
        self.actual_in_dim = in_dim + n_interact
        self.layer = _ChebyKANLinear(self.actual_in_dim, out_dim, degree)
        self.l1_lambda = 0.001

    def _add_interactions(self, x: torch.Tensor) -> torch.Tensor:
        if not self.interaction:
            return x
        idx_i, idx_j = torch.triu_indices(self.in_dim_raw, self.in_dim_raw, offset=1)
        return torch.cat([x, x[:, idx_i] * x[:, idx_j]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(self._add_interactions(x))

    def l1_reg(self) -> torch.Tensor:
        return self.l1_lambda * (self.layer.poly_weight.abs().sum() + self.layer.weight_linear.abs().sum())

    def get_var_names(self, raw_names: List[str]) -> List[str]:
        names = list(raw_names)
        if self.interaction:
            for i in range(len(raw_names)):
                for j in range(i + 1, len(raw_names)):
                    names.append(f"{raw_names[i]}*{raw_names[j]}")
        return names

    def extract_formula(self, raw_var_names: List[str], threshold: float = 1e-3) -> Dict[str, "sp.Expr"]:
        """Extract sympy parsing expressions directly from trained weights"""
        if not _HAS_SYMPY:
            raise ImportError("sympy is required for formula extraction.")
        raw_syms = [sp.Symbol(n) for n in raw_var_names]
        input_exprs: List[sp.Expr] = list(raw_syms)
        if self.interaction:
            for i in range(len(raw_syms)):
                for j in range(i + 1, len(raw_syms)):
                    input_exprs.append(raw_syms[i] * raw_syms[j])
        w_lin = self.layer.weight_linear.detach().cpu().numpy()
        bias_val = self.layer.bias.detach().cpu().numpy()
        poly_w = self.layer.poly_weight.detach().cpu().numpy()
        formulas = {}
        for out_j in range(self.layer.out_dim):
            expr = sp.Float(round(float(bias_val[out_j]), 6)) if abs(bias_val[out_j]) >= threshold else sp.Float(0)
            for i, ve in enumerate(input_exprs):
                expr += build_univariate_expr(float(w_lin[out_j, i]), poly_w[out_j, i, :], ve, threshold)
            formulas["AIDI" if self.layer.out_dim == 1 else f"AIDI_{out_j}"] = expr
        return formulas

    def get_feature_importance(self, raw_var_names: List[str]) -> pd.DataFrame:
        var_names = self.get_var_names(raw_var_names)
        imp = compute_feature_importance(
            self.layer.weight_linear.detach().cpu().numpy(),
            self.layer.poly_weight.detach().cpu().numpy())
        df = pd.DataFrame({"Feature": var_names, "Importance": imp})
        df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
        df["Importance_Norm"] = df["Importance"] / df["Importance"].sum()
        return df

    def get_activation_curves(
        self, raw_var_names: List[str],
        x_range: Tuple[float, float] = (-3.0, 3.0), n_points: int = 200,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Returns the learned activation shape of each input variable φ_i(x)"""
        var_names = self.get_var_names(raw_var_names)
        x_vals = np.linspace(x_range[0], x_range[1], n_points, dtype=np.float32)
        u = np.tanh(x_vals)
        T = [np.ones_like(u), u]
        for _ in range(2, self.layer.degree + 1):
            T.append(2 * u * T[-1] - T[-2])
        T_arr = np.stack(T, axis=-1)
        w_lin = self.layer.weight_linear.detach().cpu().numpy()
        poly_w = self.layer.poly_weight.detach().cpu().numpy()
        return {name: (x_vals, T_arr @ poly_w[0, i, :] + w_lin[0, i] * x_vals)
                for i, name in enumerate(var_names)}


# ====================== Part 3: training and evaluation ======================

def train_symbolic_kan(
    X: np.ndarray, y: np.ndarray, feature_names: List[str],
    degree: int = 4, interaction: bool = True, epochs: int = 500,
    lr: float = 1e-3, batch_size: int = 1024, l1_lambda: float = 0.001,
    device: str = "cpu", verbose: bool = True,
) -> SymbolicKAN:
    """Training SymbolicKAN to discover explicit formulas"""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) if y.ndim == 1 else torch.tensor(y, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    model = SymbolicKAN(X.shape[1], 1, degree, interaction)
    model.l1_lambda = l1_lambda
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_loss, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.mse_loss(model(xb), yb) + model.l1_reg()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            mse_val = loss.item() - model.l1_reg().item()
            total_loss += mse_val * xb.size(0); n += xb.size(0)
        scheduler.step()
        avg = total_loss / n
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                r2 = r2_score(y, model(X_t.to(device)).cpu().numpy().flatten())
            print(f"   Epoch {epoch:4d}/{epochs} | MSE: {avg:.6f} | R²: {r2:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    return model.cpu()


def evaluate_formula_kan(model: SymbolicKAN, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate SymbolicKAN fit quality"""
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
    mask = np.isfinite(pred) & np.isfinite(y)
    p, o = pred[mask], y[mask]
    return {"R2": r2_score(o, p), "RMSE": np.sqrt(mean_squared_error(o, p)),
            "MAE": np.mean(np.abs(p - o)), "Corr": np.corrcoef(p, o)[0, 1] if len(p) > 1 else np.nan}


# ====================== Part 4: HPA-UDE KAN Interpretability ======================

def extract_model_kan_info(model_path: str, device: str = "cpu") -> Dict:
    """Extract feature importance + activation curve of soil_kan / terrain_kan from trained HPA-UDE"""
    from model import HPA_UDE_Model
    model = HPA_UDE_Model(static_dim=13, dynamic_dim=9, hidden_dim=64)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    results: Dict = {}
    entries = [
        ("soil", model.hypernet.soil_kan.kan, ["Clay", "Sand", "BD", "OC", "Porosity"]),
        ("terrain", model.hypernet.terrain_kan.kan, ["Dem", "Slope"]),
    ]
    for tag, kan, feat_names in entries:
        w_lin = kan.weight_linear.detach().cpu().numpy()
        poly_w = kan.poly_weight.detach().cpu().numpy()
        results[f"{tag}_importance"] = dict(zip(feat_names, compute_feature_importance(w_lin, poly_w).tolist()))

        x_vals = np.linspace(-3, 3, 200, dtype=np.float32)
        u = np.tanh(x_vals)
        T = [np.ones_like(u), u]
        for d in range(2, kan.degree + 1):
            T.append(2 * u * T[-1] - T[-2])
        T_arr = np.stack(T, axis=-1)
        curves = {}
        for i, name in enumerate(feat_names):
            act = (T_arr @ poly_w[:, i, :].T + x_vals[:, None] * w_lin[:, i][None, :]).mean(axis=1)
            curves[name] = (x_vals, act)
        results[f"{tag}_activations"] = curves
    return results


# ====================== Part 5: Visualization ======================

def plot_activation_curves(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Learned KAN Activations", output_path: Optional[str] = None,
) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(curves); cols = min(n, 4); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for idx, (name, (x, y)) in enumerate(curves.items()):
        ax = axes[idx // cols][idx % cols]
        ax.plot(x, y, lw=2); ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(f"φ({name})", fontsize=10); ax.grid(True, alpha=0.3)
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    fig.suptitle(title, fontsize=13); plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ====================== Main process ======================

def main():
    """KAN The main process of native symbolic regression: training → evaluation → extraction formula → feature importance → interpretability"""
    import argparse
    parser = argparse.ArgumentParser(description="Native KAN Symbolic Regression")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "drought_analysis_gpu"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "symbolic_results"))
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1_lambda", type=float, default=0.001)
    parser.add_argument("--prune_threshold", type=float, default=0.01)
    parser.add_argument("--no_interaction", action="store_true")
    parser.add_argument("--max_samples", type=int, default=50000)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("🧬 Native KAN Symbolic Regression")
    print("=" * 60)

    # 1. Load data（Supports both npz and parquet formats）
    npz_path = os.path.join(args.data_dir, "pysr_dataset.npz")
    parquet_path = os.path.join(args.data_dir, "drought_indices.parquet")
    
    if os.path.exists(npz_path):
        # Use npz format（If it has been generated）
        print(f"📂 Loading from npz: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        X, y, feature_names = data["X"], data["y"], list(data["feature_names"])
    elif os.path.exists(parquet_path):
        # Use parquet format（New version of drought_indices_optimized.py output）
        print(f"📂 Loading from parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        print(f"   Available columns: {list(df.columns)}")
        
        # Build feature column（Standardized variable + SPEI + SMDI）
        var_cols = ["SM", "Pre", "PET", "LAI"]
        std_cols = [f"{v}_std" for v in var_cols if f"{v}_std" in df.columns]
        
        # Smart selection of target columns
        if "LAI_std" in df.columns:
            target_col = "LAI_std"
        elif "LAI" in df.columns:
            target_col = "LAI"
        else:
            # If there is no LAI, the first normalized column is used（Usually SM_std）
            if std_cols:
                target_col = std_cols[0]
                print(f"   ⚠️ LAI not found, using {target_col} as target")
            else:
                print(f"   ⚠️ No valid target column found in parquet file")
                return
        
        # Build valid columns（Only include columns that actually exist）
        valid_cols = [target_col]
        for col in std_cols:
            if col != target_col and col in df.columns:
                valid_cols.append(col)
        
        if "SPEI" in df.columns:
            valid_cols.append("SPEI")
        if "SMDI" in df.columns:
            valid_cols.append("SMDI")
        
        print(f"   Using columns: {valid_cols}")
        
        df_clean = df.dropna(subset=valid_cols)
        
        # subsampling（Avoid memory explosion）
        max_samples_auto = 100000
        if len(df_clean) > max_samples_auto:
            df_clean = df_clean.sample(n=max_samples_auto, random_state=42)
        
        feature_names = [c for c in valid_cols if c != target_col]
        X = df_clean[feature_names].values.astype(np.float32)
        y = df_clean[target_col].values.astype(np.float32)
        
        print(f"   Target: {target_col}")
        print(f"   Features: {feature_names}")
    else:
        print(f"⚠️ Dataset not found: {npz_path} or {parquet_path}")
        print("   Run drought_indices_optimized.py first.")
        return
    
    print(f"📂 Dataset: {X.shape[0]} samples, {len(feature_names)} features")

    if len(X) > args.max_samples:
        idx = np.random.RandomState(42).choice(len(X), args.max_samples, replace=False)
        X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # 2. train
    interaction = not args.no_interaction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_symbolic_kan(X_train, y_train, feature_names, degree=args.degree,
        interaction=interaction, epochs=args.epochs, lr=args.lr, l1_lambda=args.l1_lambda, device=device)

    # 3. Evaluate
    train_m = evaluate_formula_kan(model, X_train, y_train)
    test_m = evaluate_formula_kan(model, X_test, y_test)
    print(f"\n📊 Train R²={train_m['R2']:.4f} RMSE={train_m['RMSE']:.4f}")
    print(f"📊 Test  R²={test_m['R2']:.4f} RMSE={test_m['RMSE']:.4f}")

    # 4. Extract formula
    if _HAS_SYMPY:
        formulas = model.extract_formula(feature_names, threshold=args.prune_threshold)
        for name, expr in formulas.items():
            pruned = prune_expression(expr, args.prune_threshold)
            print(f"\n🏆 {name} (ops={sp.count_ops(pruned)}): {pruned}")
            with open(os.path.join(args.output_dir, f"{name}_formula.txt"), "w", encoding="utf-8") as f:
                f.write(f"Formula: {pruned}\nLaTeX: {sp.latex(pruned)}\n"
                        f"Train R²: {train_m['R2']:.6f}\nTest R²: {test_m['R2']:.6f}\n")

    # 5. feature importance
    imp_df = model.get_feature_importance(feature_names)
    print(f"\n📊 Feature Importance (top 10):\n{imp_df.head(10).to_string(index=False)}")
    imp_df.to_csv(os.path.join(args.output_dir, "kan_feature_importance.csv"), index=False)

    # 6. Activate curve + model save
    curves = model.get_activation_curves(feature_names)
    np.savez_compressed(os.path.join(args.output_dir, "kan_activation_curves.npz"),
        **{f"{k}_x": v[0] for k, v in curves.items()}, **{f"{k}_y": v[1] for k, v in curves.items()})
    try:
        plot_activation_curves(curves, "AIDI Activation Shapes",
            os.path.join(args.output_dir, "activation_shapes.png"))
    except Exception as e:
        print(f"   ⚠️ Plot failed: {e}")
    torch.save(model.state_dict(), os.path.join(args.output_dir, "symbolic_kan.pth"))

    # 7. HPA-UDE Interpretability（Optional）
    ckpt = args.model_path or os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
    if os.path.exists(ckpt):
        try:
            info = extract_model_kan_info(ckpt)
            for tag in ["soil", "terrain"]:
                imp = info[f"{tag}_importance"]
                mx = max(imp.values()) if imp.values() else 1.0
                print(f"\n   {tag.title()} KAN importance:")
                for feat, val in sorted(imp.items(), key=lambda x: -x[1]):
                    print(f"     {feat:10s} {val:8.3f} {'█' * int(val / mx * 20)}")
                try:
                    plot_activation_curves(info[f"{tag}_activations"],
                        f"GeoHyperNet {tag.title()} KAN", os.path.join(args.output_dir, f"model_{tag}_act.png"))
                except Exception: pass
            np.savez_compressed(os.path.join(args.output_dir, "model_kan_interpretability.npz"),
                soil_features=list(info["soil_importance"].keys()),
                soil_importance=list(info["soil_importance"].values()),
                terrain_features=list(info["terrain_importance"].keys()),
                terrain_importance=list(info["terrain_importance"].values()))
        except Exception as e:
            print(f"   ⚠️ Model extraction failed: {e}")

    print(f"\n✅ Done! Results → {args.output_dir}")


if __name__ == "__main__":
    main()
