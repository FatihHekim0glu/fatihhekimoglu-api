"""Persistence / random-walk baseline — the floor the LSTM must beat.

The persistence forecaster predicts the next-day log-return as ``0`` (equivalently
the random-walk price forecast ``P_hat_{t+1} = P_t``). On an efficient /
unit-root price process this is the error-minimizing point forecast, so it is the
honest baseline for MASE, the Diebold-Mariano test, and the ``beats_naive``
verdict. This module is pure numpy and imports no heavy dependency.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np

from lstmforecast._typing import FloatArray


class PersistenceForecaster:
    """The random-walk / persistence point forecaster (``r_hat_{t+1} = 0``).

    Stateless: there are no parameters to fit. The ``fit`` method exists only to
    mirror the model API so the walk-forward engine can treat the baseline and
    the LSTM uniformly.
    """

    def fit(self, x: FloatArray, y: FloatArray) -> PersistenceForecaster:
        """Return ``self`` unchanged (the baseline has no parameters to learn).

        Parameters
        ----------
        x:
            The train-fold sequence tensor (ignored).
        y:
            The train-fold target returns (ignored).

        Returns
        -------
        PersistenceForecaster
            ``self``, for call-chaining parity with the LSTM model.
        """
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """Predict an all-zeros next-day return vector, one per input sequence.

        Parameters
        ----------
        x:
            A ``(n_samples, look_back, n_features)`` sequence tensor; only its
            leading dimension (the sample count) is used.

        Returns
        -------
        FloatArray
            A ``(n_samples,)`` array of zeros (``r_hat_{t+1} = 0``).

        Raises
        ------
        ValidationError
            If ``x`` is not a 3-D tensor.
        """
        if x.ndim != 3:
            from lstmforecast._exceptions import ValidationError

            raise ValidationError(
                f"PersistenceForecaster.predict: x must be a 3-D "
                f"(n_samples, look_back, n_features) tensor, got ndim={x.ndim}."
            )
        return np.zeros(int(x.shape[0]), dtype="float64")


def persistence_returns(n_samples: int) -> FloatArray:
    """Return the persistence forecast vector ``zeros(n_samples)``.

    A free function for call sites (metrics, the DM test) that want the baseline
    forecast without constructing a :class:`PersistenceForecaster`.

    Parameters
    ----------
    n_samples:
        The number of forecasts to emit (``>= 0``).

    Returns
    -------
    FloatArray
        ``numpy.zeros(n_samples)``.

    Raises
    ------
    ValidationError
        If ``n_samples < 0``.
    """
    if n_samples < 0:
        from lstmforecast._exceptions import ValidationError

        raise ValidationError(f"persistence_returns: n_samples must be >= 0, got {n_samples}.")
    return np.zeros(int(n_samples), dtype="float64")
