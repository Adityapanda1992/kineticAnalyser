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
    # v = Vmax * S / (Km(1 + I/Ki) + S)
    return (vmax * s) / (km * (1 + i / ki) + s)

def uncompetitive_inhibition(s: np.ndarray, i: np.ndarray, vmax: float, km: float, ki: float) -> np.ndarray:
    """Uncompetitive Inhibition."""
    # v = Vmax * S / (Km + S(1 + I/Ki))
    return (vmax * s) / (km + s * (1 + i / ki))

def noncompetitive_inhibition(s: np.ndarray, i: np.ndarray, vmax: float, km: float, ki: float) -> np.ndarray:
    """Noncompetitive (Pure) Inhibition."""
    # v = Vmax * S / ((1 + I/Ki) * (Km + S))
    return (vmax * s) / ((1 + i / ki) * (km + s))

def mixed_inhibition(s: np.ndarray, i: np.ndarray, vmax: float, km: float, ki: float, alpha: float) -> np.ndarray:
    """Mixed Inhibition."""
    # General form: v = Vmax * S / (Km(1 + I/Ki) + S(1 + I/(alpha*Ki)))
    # Here alpha = Ki' / Ki. 
    # If alpha > 1, binding of I decreases affinity for S (Competitive-like).
    # If alpha < 1, binding of I increases affinity for S.
    # If alpha = 1, Noncompetitive.
    
    # Avoid division by zero if alpha is tiny
    if alpha < 1e-9: alpha = 1e-9
    
    term_km = km * (1 + i / ki)
    term_s = s * (1 + i / (alpha * ki))
    return (vmax * s) / (term_km + term_s)


def estimate_initial_params(s: np.ndarray, v: np.ndarray, i: Optional[np.ndarray] = None, model_type: str = 'michaelis_menten') -> List[float]:
    """
    Estimates initial params.
    """
    try:
        # 1. Basic MM est from data where I=0 (or all data if I not provided)
        mask = (v > 1e-9) & (s > 1e-9)
        if i is not None:
             # Try to use only uninhibited data for Vmax/Km guess
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
        
        # Bounds logic
        vmax_est = max(vmax_est, np.max(v) * 0.5)
        km_est = max(km_est, 1e-6)
        
        # 2. Estimate Ki if needed
        if model_type == 'michaelis_menten':
            return [vmax_est, km_est]
            
        elif model_type == 'substrate_inhibition':
            # Ki is usually high
            ki_est = np.max(s) * 2
            return [vmax_est, km_est, ki_est]
            
        elif model_type in ['competitive', 'uncompetitive', 'noncompetitive', 'mixed']:
            # Rough guess for Ki: often comparable to Km or concentrations used
            ki_est = np.median(i[i > 0]) if i is not None and np.any(i > 0) else 1.0
            
            if model_type == 'mixed':
                return [vmax_est, km_est, ki_est, 1.0] # alpha=1 (Noncomp start)
            else:
                return [vmax_est, km_est, ki_est]
                
        return [vmax_est, km_est]

    except Exception:
        # Absolute fallback
        if model_type == 'mixed':
             return [np.max(v), np.median(s), 1.0, 1.0]
        elif model_type == 'michaelis_menten':
             return [np.max(v), np.median(s)]
        else:
             return [np.max(v), np.median(s), 1.0]


def fit_data(concentrations: List[float], 
             rates: List[float], 
             inhibitors: Optional[List[float]] = None,
             model_type: str = 'michaelis_menten',
             weighting: Optional[str] = None, 
             robust: bool = False) -> Optional[Dict]:
    """
    Fits kinetic data.
    """

    x_data = np.array(concentrations, dtype=float)
    y_data = np.array(rates, dtype=float)
    i_data = np.array(inhibitors, dtype=float) if inhibitors is not None else None

    # ---- Validate Data ----
    if len(x_data) != len(y_data): return None
    if i_data is not None and len(i_data) != len(x_data): return None
    
    # Requirement: Inhibition models need inhibitor data
    if model_type in ['competitive', 'uncompetitive', 'noncompetitive', 'mixed'] and i_data is None:
        return None

    # ---- Initial Guesses ----
    p0 = estimate_initial_params(x_data, y_data, i_data, model_type)

    # ---- Weights ----
    if weighting is None:
        sigma = np.ones_like(y_data)
    elif weighting == "1/y":
        sigma = np.sqrt(np.abs(y_data))
        sigma[sigma < 1e-9] = 1.0 
    elif weighting == "1/y2":
        sigma = np.abs(y_data)
        sigma[sigma < 1e-9] = 1.0
    else:
        sigma = np.ones_like(y_data)

    # ---- Residuals ----
    def residuals(params, x, y, i_conc, w_sigma):
        model_v = np.zeros_like(x)
        
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
            
        return (y - model_v) / w_sigma

    # ---- Bounds ----
    bounds_lower = [0] * len(p0)
    bounds_upper = [np.inf] * len(p0)

    # ---- Fit ----
    try:
        result = least_squares(
            residuals,
            p0,
            args=(x_data, y_data, i_data, sigma),
            bounds=(bounds_lower, bounds_upper),
            loss='huber' if robust else 'linear',
            max_nfev=2000
        )
    except Exception as e:
        print(f"Fitting error: {e}")
        return None

    if not result.success:
        return None

    params = result.x
    
    # ---- Statistics ----
    J = result.jac
    rss = np.sum(result.fun**2)
    n = len(y_data)
    p = len(params)
    
    # Degrees of freedom
    dof = n - p
    if dof <= 0:
        # Cannot estimate error with zero or negative degrees of freedom
        perr = np.full_like(params, np.nan)
    else:
        # Reduced chi-square (residual variance)
        residual_var = rss / dof
        
        # If fit is numerically 'perfect' (unlikely for real data), 
        # we don't want to report 0 error unless it's truly exact.
        if residual_var < 1e-18:
            residual_var = 0.0

        try:
            # Use SVD to check for singular values
            # This helps identify which specific parameters are unconstrained
            U, s, Vh = np.linalg.svd(J, full_matrices=False)
            
            # Threshold for singularity (standard heuristic)
            threshold = np.max(s) * max(J.shape) * np.finfo(s.dtype).eps
            
            # Identify columns of J (parameters) that are effectively null
            # or linearly dependent. Better yet, use the pseudo-inverse 
            # and then check the diagonal of the resulting covariance.
            jtj_inv = np.linalg.pinv(J.T @ J, rcond=1e-12)
            cov = jtj_inv * residual_var
            
            # A diagonal element of jtj_inv being near zero after pinv 
            # usually means that parameter was suppressed because its singular value was too small.
            # We explicitly label these as NaN.
            diag_jtj = np.diag(jtj_inv)
            perr = []
            for i in range(len(params)):
                # If the singular value associated with this parameter is effectively suppressed 
                # or if residual_var is effectively zero, we can't reliably report precision.
                if diag_jtj[i] < 1e-20: 
                    perr.append(np.nan)
                else:
                    perr.append(np.sqrt(max(cov[i, i], 0)))
            perr = np.array(perr)

        except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
            perr = np.full_like(params, np.nan)

    # ---- Results Packet ----
    results = {
        'model': model_type,
        'fitted_params': params.tolist(),
        'param_errors': perr.tolist(),
        'aic': n * np.log(rss/n) + 2*p if rss > 0 else -np.inf,
        'rss': rss,
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
    
    # ---- R-squared & Model Generation ----
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

    ss_res = np.sum((y_data - y_model)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    results['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
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
        # For inhibition, we generate curves for each UNIQUE inhibitor concentration present in data
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
