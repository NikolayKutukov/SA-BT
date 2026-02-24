"""SurvTRACE wrapper — transformer-based survival model (single + competing risks)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import SurvivalModel
from ._survtrace.config import default_config
from ._survtrace.utils import LabelTransform


class SurvTRACEModel(SurvivalModel):
    """Transformer-based survival model (SurvTRACE).

    Wraps the vendored SurvTRACE code (MIT license) for both
    single-event and competing-risks settings.

    The model treats all features as numerical by default.  If the
    ``n_categorical`` parameter is set, the first ``n_categorical``
    columns of X are treated as integer-encoded categorical features
    (used as embedding look-ups), and the rest as numerical.
    """

    name = "SurvTRACE"
    supports_competing_risks = True

    def __init__(
        self,
        n_categorical: int = 0,
        vocab_size: int = 64,
        hidden_size: int = 16,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 2,
        intermediate_size: int = 64,
        dropout: float = 0.0,
        num_durations: int = 100,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        val_fraction: float = 0.2,
        early_stop_patience: int = 10,
    ) -> None:
        self._n_categorical = n_categorical
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._num_hidden_layers = num_hidden_layers
        self._num_attention_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._dropout = dropout
        self._num_durations = num_durations
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._weight_decay = weight_decay
        self._val_fraction = val_fraction
        self._early_stop_patience = early_stop_patience

        self._model = None
        self._label_transform = None
        self._competing = False
        self._n_risks = 1
        self._config = None

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "SurvTRACEModel":
        import torch
        from ._survtrace.model import SurvTraceSingle, SurvTraceMulti
        from ._survtrace.train_utils import Trainer

        seed = kwargs.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Detect competing risks
        unique_events = sorted(set(int(e) for e in E if e > 0))
        self._competing = len(unique_events) > 1
        self._n_risks = len(unique_events) if self._competing else 1

        n, p = X.shape
        n_cat = self._n_categorical
        n_num = p - n_cat

        # Discretise time
        cuts = np.linspace(T.min(), T.max(), self._num_durations + 1)
        self._label_transform = LabelTransform(cuts)
        self._label_transform.fit(T.astype("float64"), E.astype("int64"))
        out_feature = self._label_transform.out_features

        # Build config
        self._config = default_config()
        self._config.update({
            "num_feature": p,
            "num_numerical_feature": n_num,
            "num_categorical_feature": n_cat,
            "vocab_size": max(self._vocab_size, int(X[:, :n_cat].max()) + 1) if n_cat > 0 else 2,
            "hidden_size": self._hidden_size,
            "num_hidden_layers": self._num_hidden_layers,
            "num_attention_heads": self._num_attention_heads,
            "intermediate_size": self._intermediate_size,
            "out_feature": out_feature,
            "num_event": self._n_risks,
            "early_stop_patience": self._early_stop_patience,
            "duration_index": self._label_transform.cuts,  # full cut points (predict_surv pads at start)
            "hidden_dropout_prob": self._dropout,
            "attention_probs_dropout_prob": self._dropout,
            "checkpoint": kwargs.get("checkpoint", "./checkpoints/survtrace.pt"),
        })

        # Build model
        if self._competing:
            self._model = SurvTraceMulti(self._config)
        else:
            self._model = SurvTraceSingle(self._config)

        # Prepare data as DataFrames (SurvTRACE Trainer expects them)
        # Columns: [cat_0, ..., cat_k, num_0, ..., num_m]
        # Reorder X: categorical first, then numerical
        if n_cat > 0:
            X_ordered = np.column_stack([X[:, :n_cat], X[:, n_cat:]])
        else:
            X_ordered = X.copy()

        # Train / val split
        n_val = int(n * self._val_fraction)
        perm = np.random.permutation(n)
        idx_val, idx_train = perm[:n_val], perm[n_val:]

        X_train, X_val = X_ordered[idx_train], X_ordered[idx_val]
        T_train, T_val = T[idx_train], T[idx_val]
        E_train, E_val = E[idx_train], E[idx_val]

        # Transform labels
        df_x_train = pd.DataFrame(X_train, columns=[f"f_{i}" for i in range(p)])
        df_x_val = pd.DataFrame(X_val, columns=[f"f_{i}" for i in range(p)])

        if self._competing:
            df_y_train = self._make_multi_event_labels(T_train, E_train, unique_events)
            df_y_val = self._make_multi_event_labels(T_val, E_val, unique_events)
        else:
            df_y_train = self._make_single_event_labels(T_train, E_train)
            df_y_val = self._make_single_event_labels(T_val, E_val)

        trainer = Trainer(self._model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(
                train_set=(df_x_train, df_y_train),
                val_set=(df_x_val, df_y_val),
                batch_size=self._batch_size,
                epochs=self._epochs,
                learning_rate=self._lr,
                weight_decay=self._weight_decay,
                verbose=False,
            )

        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        df = self._make_input_df(X)
        surv_df = self._model.predict_surv_df(df, batch_size=256)
        # Risk = negative mean survival
        return -surv_df.values.mean(axis=0)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        df = self._make_input_df(X)
        surv_df = self._model.predict_surv_df(df, batch_size=256)
        return self._interpolate_surv(surv_df, times)

    def predict_cumulative_incidence(
        self, X: np.ndarray, times: np.ndarray, cause: int
    ) -> np.ndarray:
        if not self._competing:
            raise NotImplementedError("CIF requires competing risks mode.")

        df = self._make_input_df(X)
        # cause is 1-indexed; SurvTRACE multi uses 0-indexed event
        event_idx = cause - 1

        # Get cause-specific survival, then CIF = 1 - S_k(t)
        surv_df = self._model.predict_surv_df(df, batch_size=256, event=event_idx)
        surv = self._interpolate_surv(surv_df, times)
        return np.clip(1.0 - surv, 0.0, 1.0)

    # ── Internal helpers ─────────────────────────────────────────────

    def _make_input_df(self, X: np.ndarray) -> pd.DataFrame:
        """Create DataFrame in the column order SurvTRACE expects."""
        n_cat = self._n_categorical
        p = X.shape[1]
        if n_cat > 0:
            X_ordered = np.column_stack([X[:, :n_cat], X[:, n_cat:]])
        else:
            X_ordered = X
        return pd.DataFrame(X_ordered, columns=[f"f_{i}" for i in range(p)])

    def _make_single_event_labels(
        self, T: np.ndarray, E: np.ndarray
    ) -> pd.DataFrame:
        """Create label DataFrame for single-event: [duration, event, proportion]."""
        idx_dur, events, t_frac = self._label_transform.transform(
            T.astype("float64"), E.astype("int64")
        )
        return pd.DataFrame({
            "duration": idx_dur,
            "event": events,
            "proportion": t_frac,
        })

    def _make_multi_event_labels(
        self, T: np.ndarray, E: np.ndarray, causes: list[int]
    ) -> pd.DataFrame:
        """Create label DataFrame for competing risks.

        Columns: duration, event_0, event_1, ..., proportion
        """
        idx_dur, _, t_frac = self._label_transform.transform(
            T.astype("float64"), E.astype("int64")
        )
        data = {"duration": idx_dur, "proportion": t_frac}
        for i, cause_k in enumerate(causes):
            # Binary indicator for cause k
            data[f"event_{i}"] = (E == cause_k).astype("float32")
        return pd.DataFrame(data)

    @staticmethod
    def _interpolate_surv(surv_df, times: np.ndarray) -> np.ndarray:
        index = surv_df.index.values.astype(float)
        values = surv_df.values  # (n_time_grid, n_subjects)
        n_subjects = values.shape[1]
        out = np.zeros((n_subjects, len(times)))
        for j in range(n_subjects):
            out[j] = np.interp(times, index, values[:, j], left=1.0, right=0.0)
        return np.clip(out, 0.0, 1.0)
