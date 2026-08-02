import numpy as np
from scipy.optimize import least_squares
from typing import Optional, Dict, List, Tuple, Union

# --- Model Equations ---

def michaelis_menten(s: np.ndarray, vmax: float, km: float) -> np.ndarray:
    """Standard Michaelis-Menten equation."""
    return (vmax * s) / (km + s)

def substrate_inhibition(s: np.ndarray, vmax: float, km: float, ki: float) -> np.ndarray:
    """Substrate Inhibition (Haldane) equation."""
    return (vmax * s) / (km + s + (s**2 / ki))

def competitive_inhibition(s: np.ndarray, i: np.ndarray, vmax: float, km: float, ki: float) -> np.ndarray:
    """Competitive Inhibition."""
    return (vmax * s) / (km * (1 + i / ki) + s)

def uncompetitive_inhibition(s: np.ndarray, i: np.ndarray, vmax: float, km: float, ki: float) -> np.ndarray:
    """Uncompetitive Inhibition."""
    return (vmax * s) / (km + s * (1 + i / ki))

def noncompetitive_inhibition(s: np.ndarray, i: np.ndarray, vmax: float, km: float, ki: float) -> np.ndarray:
    """Noncompetitive (Pure) Inhibition."""
    return (vmax * s) / ((1 + i / ki) * (km + s))

def mixed_inhibition(s: np.ndarray, i: np.ndarray, vmax: float, km: float, ki: float, alpha: float) -> np.ndarray:
    """Mixed Inhibition."""
    if alpha < 1e-12: alpha = 1e-12
    term_km = km * (1 + i / ki)
    term_s = s * (1 + i / (alpha * ki))
    return (vmax * s) / (term_km + term_s)


def estimate_initial_params(s: np.ndarray, v: np.ndarray, i: Optional[np.ndarray] = None, model_type: str = 'michaelis_menten') -> List[float]:
    """
    Estimates initial parameter values for non-linear regression.
    """
    try:
        mask = (v > 1e-9) & (s > 1e-9)
        if i is not None:
             mask_zero_i = mask & (i < 1e-9)
             if np.sum(mask_zero_i) >= 2:
                 s_for_est = s[mask_zero_i]
                 v_for_est = v[mask_zero_i]
             else:
                 s_for_est = s[mask]
                 v_for_est = v[mask]
        else:
             s_for_est = s[mask]
             v_for_est = v[mask]

        if len(v_for_est) < 2:
             vmax_est = np.max(v)
             km_est = np.median(s)
        else:
            # Hanes-Woolf: S/v = (1/Vmax)S + (Km/Vmax)
            y_lin = s_for_est / v_for_est
            slope, intercept = np.polyfit(s_for_est, y_lin, 1)
            
            if slope > 1e-9:
                vmax_est = 1.0 / slope
                km_est = intercept * vmax_est
            else:
                vmax_est = np.max(v)
                km_est = np.median(s)
        
        vmax_est = max(vmax_est, np.max(v) * 0.5)
        km_est = max(km_est, 1e-6)
        
        if model_type == 'michaelis_menten':
            return [vmax_est, km_est]
            
        elif model_type == 'substrate_inhibition':
            ki_est = np.max(s) * 2.0
            return [vmax_est, km_est, ki_est]
            
        elif model_type in ['competitive', 'uncompetitive', 'noncompetitive', 'mixed']:
            ki_est = np.median(i[i > 0]) if i is not None and np.any(i > 0) else 1.0
            
            if model_type == 'mixed':
                return [vmax_est, km_est, ki_est, 1.0] # alpha=1.0 initial guess
            else:
                return [vmax_est, km_est, ki_est]
                
        return [vmax_est, km_est]

    except Exception:
        if model_type == 'mixed':
             return [np.max(v), np.median(s), 1.0, 1.0]
        elif model_type == 'michaelis_menten':
             return [np.max(v), np.median(s)]
        else:
             return [np.max(v), np.median(s), 1.0]


def compute_weights(y_data: np.ndarray, weighting: Optional[str] = None) -> np.ndarray:
    """
    Computes standard deviations (sigma) for weighting with a continuous relative noise floor.
    Prevents artificial step-discontinuities and zero division near base rate levels.
    """
    if weighting is None or weighting == "None":
        return np.ones_like(y_data)
    
    # Continuous noise floor based on 1% of mean rate magnitude
    noise_floor = max(1e-6, 0.01 * np.mean(np.abs(y_data)))
    y_clamped = np.maximum(np.abs(y_data), noise_floor)
    
    if weighting == "1/y":
        return np.sqrt(y_clamped)
    elif weighting == "1/y2":
        return y_clamped
    else:
        return np.ones_like(y_data)


def fit_data(concentrations: List[float], 
             rates: List[float], 
             inhibitors: Optional[List[float]] = None,
             model_type: str = 'michaelis_menten',
             weighting: Optional[str] = None, 
             robust: bool = False) -> Optional[Dict]:
    """
    Fits kinetic data using non-linear least squares optimization with high scientific accuracy.
    Includes:
    - Strictly positive parameter bounds (prevents division by zero)
    - Relative noise-floored weighting schemes (prevents near-zero rate exploding weights)
    - Correct unweighted vs weighted RSS and R² evaluation
    - Small-Sample Corrected Akaike Information Criterion (AICc)
    """

    x_data = np.array(concentrations, dtype=float)
    y_data = np.array(rates, dtype=float)
    i_data = np.array(inhibitors, dtype=float) if inhibitors is not None else None

    # ---- Validate Data ----
    if len(x_data) != len(y_data): return None
    if i_data is not None and len(i_data) != len(x_data): return None
    
    # Requirement: Inhibition models need inhibitor data with variation
    if model_type in ['competitive', 'uncompetitive', 'noncompetitive', 'mixed']:
        if i_data is None:
            return None
        if np.std(i_data) < 1e-9:
            # Inhibitor concentration does not vary; Ki is unidentifiable
            return None

    # ---- Initial Guesses ----
    p0 = estimate_initial_params(x_data, y_data, i_data, model_type)

    # ---- Weights ----
    sigma = compute_weights(y_data, weighting)

    # ---- Residuals Function ----
    def residuals(params, x, y, i_conc, w_sigma):
        if model_type == 'michaelis_menten':
            model_v = michaelis_menten(x, params[0], params[1])
        elif model_type == 'substrate_inhibition':
            model_v = substrate_inhibition(x, params[0], params[1], params[2])
        elif model_type == 'competitive':
            model_v = competitive_inhibition(x, i_conc, params[0], params[1], params[2])
        elif model_type == 'uncompetitive':
            model_v = uncompetitive_inhibition(x, i_conc, params[0], params[1], params[2])
        elif model_type == 'noncompetitive':
            model_v = noncompetitive_inhibition(x, i_conc, params[0], params[1], params[2])
        elif model_type == 'mixed':
            model_v = mixed_inhibition(x, i_conc, params[0], params[1], params[2], params[3])
        else:
            model_v = michaelis_menten(x, params[0], params[1])
            
        return (y - model_v) / w_sigma

    # ---- Strictly Positive Bounds ----
    # Prevents parameters touching 0.0 which leads to 1/0 division in rate laws
    bounds_lower = [1e-12] * len(p0)
    bounds_upper = [np.inf] * len(p0)

    # ---- Fit ----
    try:
        result = least_squares(
            residuals,
            p0,
            args=(x_data, y_data, i_data, sigma),
            bounds=(bounds_lower, bounds_upper),
            loss='huber' if robust else 'linear',
            max_nfev=3000
        )
    except Exception as e:
        print(f"Fitting error: {e}")
        return None

    if not result.success:
        return None

    params = result.x
    
    # ---- Model Evaluation ----
    if model_type == 'michaelis_menten':
        y_model = michaelis_menten(x_data, *params)
    elif model_type == 'substrate_inhibition':
        y_model = substrate_inhibition(x_data, *params)
    elif model_type == 'competitive':
        y_model = competitive_inhibition(x_data, i_data, *params)
    elif model_type == 'uncompetitive':
        y_model = uncompetitive_inhibition(x_data, i_data, *params)
    elif model_type == 'noncompetitive':
        y_model = noncompetitive_inhibition(x_data, i_data, *params)
    elif model_type == 'mixed':
        y_model = mixed_inhibition(x_data, i_data, *params)

    n = len(y_data)
    p = len(params)
    dof = n - p

    # Residual Sum of Squares (Unweighted)
    ss_res_unweighted = np.sum((y_data - y_model)**2)
    ss_tot_unweighted = np.sum((y_data - np.mean(y_data))**2)
    r2_unweighted = 1.0 - (ss_res_unweighted / ss_tot_unweighted) if ss_tot_unweighted > 0 else 0.0

    # Weighted Residual Sum of Squares (Chi-square)
    chi_square = np.sum(result.fun**2)

    # Weighted R-squared
    w = 1.0 / (sigma**2)
    y_bar_w = np.sum(w * y_data) / np.sum(w)
    ss_tot_w = np.sum(w * (y_data - y_bar_w)**2)
    r2_weighted = 1.0 - (chi_square / ss_tot_w) if ss_tot_w > 0 else 0.0

    # ---- Statistics & Parameter Uncertainties ----
    if dof <= 0:
        perr = np.full_like(params, np.nan)
    else:
        residual_var = chi_square / dof
        if residual_var < 1e-18:
            residual_var = 0.0

        try:
            J = result.jac
            jtj_inv = np.linalg.pinv(J.T @ J, rcond=1e-12)
            cov = jtj_inv * residual_var
            diag_jtj = np.diag(jtj_inv)
            perr = []
            for i in range(len(params)):
                if diag_jtj[i] < 1e-20: 
                    perr.append(np.nan)
                else:
                    perr.append(np.sqrt(max(cov[i, i], 0)))
            perr = np.array(perr)

        except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
            perr = np.full_like(params, np.nan)

    # ---- Akaike Information Criterion (AIC & AICc) ----
    if weighting is not None and weighting != "None":
        # Weighted log-likelihood AIC
        aic = n * np.log(chi_square / n) + 2 * np.sum(np.log(sigma)) + 2 * p
    else:
        aic = n * np.log(ss_res_unweighted / n) + 2 * p if ss_res_unweighted > 0 else -np.inf

    # Small-sample corrected AIC (AICc)
    if n - p - 1 > 0:
        aicc = aic + (2 * p * (p + 1)) / (n - p - 1)
    else:
        aicc = aic

    # ---- Results Packet ----
    results = {
        'model': model_type,
        'fitted_params': params.tolist(),
        'param_errors': perr.tolist(),
        'aic': aic,
        'aicc': aicc,
        'rss': ss_res_unweighted,
        'chi_square': chi_square,
        'r_squared': r2_weighted if (weighting and weighting != "None") else r2_unweighted,
        'r_squared_unweighted': r2_unweighted,
        'r_squared_weighted': r2_weighted,
        'dof': dof
    }

    # Map parameters by name
    names = []
    if model_type == 'michaelis_menten':
        names = ['vmax', 'km']
    elif model_type == 'substrate_inhibition':
        names = ['vmax', 'km', 'ki']
    elif model_type in ['competitive', 'uncompetitive', 'noncompetitive']:
        names = ['vmax', 'km', 'ki']
    elif model_type == 'mixed':
        names = ['vmax', 'km', 'ki', 'alpha']

    for idx, name in enumerate(names):
        try:
             results[name] = params[idx]
             results[f"{name}_err"] = perr[idx]
        except IndexError:
             pass
        
    # Extra for Mixed: calculate Ki_prime
    if model_type == 'mixed':
        results['ki_prime'] = results['ki'] * results['alpha']
    
    results['residuals'] = (y_data - y_model).tolist()

    # ---- Smooth Curves for Plotting ----
    x_smooth = np.linspace(0, np.max(x_data)*1.1, 100)
    
    if model_type in ['michaelis_menten', 'substrate_inhibition']:
        if model_type == 'michaelis_menten':
            y_smooth = michaelis_menten(x_smooth, *params)
        else:
            y_smooth = substrate_inhibition(x_smooth, *params)
        results['fitted_curve'] = (x_smooth.tolist(), y_smooth.tolist())
        
    else:
        unique_i = np.unique(i_data)
        unique_i.sort()
        curves = {}
        for i_val in unique_i:
            i_arr = np.full_like(x_smooth, i_val)
            if model_type == 'competitive':
                y_s = competitive_inhibition(x_smooth, i_arr, *params)
            elif model_type == 'uncompetitive':
                y_s = uncompetitive_inhibition(x_smooth, i_arr, *params)
            elif model_type == 'noncompetitive':
                y_s = noncompetitive_inhibition(x_smooth, i_arr, *params)
            elif model_type == 'mixed':
                y_s = mixed_inhibition(x_smooth, i_arr, *params)
            curves[float(i_val)] = (x_smooth.tolist(), y_s.tolist())
            
        results['fitted_curves'] = curves

    return results

