"""Configuration dataclasses for runs and sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

DEFAULT_TICKERS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "GLD", "TLT")
DEFAULT_COST_BPS_GRID: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
TRADING_DAYS_PER_YEAR: int = 252


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Parameters for a single MA crossover strategy."""

    fast_window: int
    slow_window: int

    def __post_init__(self) -> None:
        if self.fast_window < 1 or self.slow_window < 1:
            raise ValueError("MA windows must be positive integers")
        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"fast_window ({self.fast_window}) must be strictly less than "
                f"slow_window ({self.slow_window})"
            )


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Transaction cost configuration in basis points (1 bp = 0.01%)."""

    per_side_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.per_side_bps < 0:
            raise ValueError(f"per_side_bps must be >= 0, got {self.per_side_bps}")
        if self.per_side_bps > 1000:
            raise ValueError(f"per_side_bps={self.per_side_bps} looks wrong (units are bps not %)")

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * self.per_side_bps

    @property
    def per_side_fraction(self) -> float:
        return self.per_side_bps / 10_000.0


@dataclass(frozen=True, slots=True)
class RunConfig:
    """End-to-end backtest run parameters."""

    ticker: str
    start: date
    end: date
    strategy: StrategyConfig
    cost: CostConfig
    initial_cash: float = 100_000.0
    risk_free_annual: float = 0.0

    def __post_init__(self) -> None:
        if not self.ticker.strip():
            raise ValueError("ticker must be a non-empty string")
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be after start ({self.start})")
        if self.initial_cash <= 0:
            raise ValueError(f"initial_cash must be > 0, got {self.initial_cash}")


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Parameter grid for a sweep across (fast, slow) pairs."""

    fast_windows: tuple[int, ...]
    slow_windows: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.fast_windows or not self.slow_windows:
            raise ValueError("fast_windows and slow_windows must each be non-empty")
        if not any(f < s for f in self.fast_windows for s in self.slow_windows):
            raise ValueError(
                "SweepConfig produces an empty grid: no fast<slow pair exists "
                f"between fast_windows={self.fast_windows} and slow_windows={self.slow_windows}"
            )

    def grid(self) -> list[StrategyConfig]:
        return [
            StrategyConfig(fast_window=f, slow_window=s)
            for f in self.fast_windows
            for s in self.slow_windows
            if f < s
        ]

    @property
    def size(self) -> int:
        return len(self.grid())


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Anchored walk-forward configuration.

    Anchored = expanding train window with a fixed-length out-of-sample step.
    ``step_years`` must be >= ``test_years`` so OOS slices are non-overlapping
    and the concatenated OOS return series is a clean contiguous sample.
    """

    train_years: int = 5
    test_years: int = 1
    step_years: int = 1
    selection_metric: str = "sharpe"
    use_neighbourhood_tiebreak: bool = True

    def __post_init__(self) -> None:
        if self.train_years < 1:
            raise ValueError(f"train_years must be >= 1, got {self.train_years}")
        if self.test_years < 1:
            raise ValueError(f"test_years must be >= 1, got {self.test_years}")
        if self.step_years < 1:
            raise ValueError(f"step_years must be >= 1, got {self.step_years}")
        if self.step_years < self.test_years:
            raise ValueError(
                f"step_years ({self.step_years}) must be >= test_years "
                f"({self.test_years}) to keep OOS slices non-overlapping"
            )


DEFAULT_SWEEP: SweepConfig = SweepConfig(
    fast_windows=tuple(range(5, 105, 5)),
    slow_windows=tuple(range(20, 220, 10)),
)
