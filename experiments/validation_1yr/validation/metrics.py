"""Agreement between the aggregated simulation and the real series."""
import numpy as np
import pandas as pd

from .normalize import peak_normalize


def compare(agg: pd.DataFrame, real: pd.Series) -> dict:
    """agg: columns mean/q05/q95 indexed by date; real: same index, may contain NaN."""
    ok = real.notna() & agg["mean"].notna()
    if ok.sum() < 3:
        return {"n_days": int(ok.sum())}
    sim, r = agg["mean"][ok], real[ok]
    err = sim - r
    sim_n, r_n = peak_normalize(sim), peak_normalize(r)
    return {
        "n_days": int(ok.sum()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
        "rmse_peak_normalized": float(np.sqrt(((sim_n - r_n) ** 2).mean())),
        "pearson_r": float(np.corrcoef(sim, r)[0, 1]),
        "peak_ratio_sim_over_real": float(sim.max() / r.max()) if r.max() else np.nan,
        "peak_day_offset": int((sim.idxmax() - r.idxmax()).days),
        "coverage_real_in_5_95_band": float(((r >= agg["q05"][ok]) & (r <= agg["q95"][ok])).mean()),
    }
