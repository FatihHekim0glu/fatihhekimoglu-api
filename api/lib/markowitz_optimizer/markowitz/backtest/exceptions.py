"""Exception hierarchy for the ``markowitz.backtest`` subpackage."""

from __future__ import annotations


class BacktestError(Exception):
    """Base class for all errors raised inside ``markowitz.backtest``."""


class InsufficientHistoryError(BacktestError):
    """Raised when fewer observations are available than the rolling window."""


class AlignmentError(BacktestError):
    """Raised when the universe of assets is inconsistent across inputs."""


class MissingMarketCapsError(BacktestError):
    """Raised when a strategy that requires market caps cannot find them."""


class DegenerateWindowError(BacktestError):
    """Raised when a rolling window is rank-deficient or non-finite."""
