"""Utility functions for SurvTRACE (vendored + cleaned up)."""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from pycox.preprocessing.discretization import (
    make_cuts,
    IdxDiscUnknownC,
    _values_if_series,
    DiscretizeUnknownC,
    Duration2Idx,
)


class LabelTransform:
    """Defining time intervals (``cuts``) needed for the PC-Hazard method.

    Parameters
    ----------
    cuts : int or array-like
        Number of cuts or actual cut points.
    scheme : str
        Discretization scheme: 'equidistant' or 'quantiles'.
    min_ : float
        Starting duration.
    """

    def __init__(self, cuts, scheme="equidistant", min_=0.0, dtype=None):
        self._cuts = cuts
        self._scheme = scheme
        self._min = min_
        self._dtype_init = dtype
        self._predefined_cuts = False
        self.cuts = None
        if hasattr(cuts, "__iter__"):
            if isinstance(cuts, list):
                cuts = np.array(cuts)
            self.cuts = cuts
            self.idu = IdxDiscUnknownC(self.cuts)
            assert dtype is None, "Need `dtype` to be `None` for specified cuts"
            self._dtype = type(self.cuts[0])
            self._dtype_init = self._dtype
            self._predefined_cuts = True
        else:
            self._cuts += 1

    def fit(self, durations, events):
        self._dtype = self._dtype_init
        if self._dtype is None:
            if isinstance(durations[0], np.floating):
                self._dtype = durations.dtype
            else:
                self._dtype = np.dtype("float64")
        durations = durations.astype(self._dtype)
        self.duc = DiscretizeUnknownC(self.cuts, right_censor=True, censor_side="right")
        self.di = Duration2Idx(self.cuts)
        return self

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        return self.transform(durations, events)

    def transform(self, durations, events):
        durations = _values_if_series(durations)
        durations = durations.astype(self._dtype)
        events = _values_if_series(events)
        dur_disc, events = self.duc.transform(durations, events)
        idx_durations = self.di.transform(dur_disc)
        cut_diff = np.diff(self.cuts)
        assert (cut_diff > 0).all(), "Cuts are not unique."
        t_frac = 1.0 - (dur_disc - durations) / cut_diff[idx_durations - 1]
        if idx_durations.min() == 0:
            warnings.warn(
                "Got event/censoring at start time. "
                "Should be removed! It is set s.t. it has no contribution to loss."
            )
            t_frac[idx_durations == 0] = 0
            events[idx_durations == 0] = 0
        idx_durations = idx_durations - 1
        idx_durations[idx_durations < 0] = 0
        return (
            idx_durations.astype("int64"),
            events.astype("float32"),
            t_frac.astype("float32"),
        )

    @property
    def out_features(self):
        if self.cuts is None:
            raise ValueError("Need to call `fit` before this is accessible.")
        return len(self.cuts) - 1


def pad_col(input, val=0, where="end"):
    """Add a column of ``val`` at the start or end of a 2-D tensor."""
    if len(input.shape) != 2:
        raise ValueError("Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == "end":
        return torch.cat([input, pad], dim=1)
    elif where == "start":
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")


def log_softplus(input, threshold=-15.0):
    """Equivalent to ``F.softplus(input).log()`` with numerical stability."""
    output = input.clone()
    above = input >= threshold
    output[above] = F.softplus(input[above]).log()
    return output


def cumsum_reverse(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if dim != 1:
        raise NotImplementedError
    input = input.sum(1, keepdim=True) - pad_col(input, where="start").cumsum(1)
    return input[:, :-1]


def set_random_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
