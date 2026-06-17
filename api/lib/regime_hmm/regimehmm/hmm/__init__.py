"""Gaussian Hidden Markov Model kernel.

The genuinely new code in this project: a pure numpy/scipy Gaussian HMM with a
log-space forward-backward pass, Baum-Welch EM with seeded restarts, the Viterbi
MAP path (EDA only), and — critically — the ONLINE forward filter that produces
the only tradable, no-lookahead regime posterior.

Importing this subpackage has no side effects.
"""

from __future__ import annotations
