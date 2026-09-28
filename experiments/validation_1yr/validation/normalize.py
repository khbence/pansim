"""Transforms that make simulated and real series comparable."""
import pandas as pd


def apply_transform(s: pd.Series, transform: str) -> pd.Series:
    if transform == "level":
        return s
    if transform == "cumsum":
        return s.cumsum()
    if transform == "roll7":
        return s.rolling(7, center=True, min_periods=1).mean()
    raise ValueError(f"Unknown transform {transform!r}")


def real_transform(s: pd.Series, transform: str) -> pd.Series:
    """Real series already are cumulative/levels where relevant, so only smooth for roll7."""
    if transform == "roll7":
        return s.rolling(7, center=True, min_periods=1).mean()
    return s


def rescale_real(s: pd.Series, sim_population: float, hungary_population: float) -> pd.Series:
    """Real counts (all of Hungary) -> counts for a population of the simulated size."""
    return s * (sim_population / hungary_population)


def peak_normalize(s: pd.Series) -> pd.Series:
    """Divide by the series' own maximum (shape comparison, scale-free)."""
    peak = s.max()
    return s / peak if peak and peak == peak else s * float("nan")
