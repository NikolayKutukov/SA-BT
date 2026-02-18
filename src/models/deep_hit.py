"""DeepHit wrapper — discrete-time deep model (single event + competing risks)."""

from __future__ import annotations

import numpy as np

from .base import SurvivalModel


class DeepHitModel(SurvivalModel):
    """DeepHit: discrete-time survival model without PH assumption.

    Uses ``pycox.models.DeepHitSingle`` for single-event setups and
    ``pycox.models.DeepHit`` for competing risks (Setup 4).

    The model discretises time into bins and predicts a probability
    mass function over the time grid — no proportional hazards
    assumption is required.
    """

    name = "DeepHit"
    supports_competing_risks = True

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        val_fraction: float = 0.2,
        patience: int = 10,
        num_durations: int = 100,
        alpha: float = 0.2,
        sigma: float = 0.1,
    ) -> None:
        self._hidden_layers = hidden_layers or [64, 64]
        self._dropout = dropout
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._val_fraction = val_fraction
        self._patience = patience
        self._num_durations = num_durations
        self._alpha = alpha
        self._sigma = sigma

        self._model = None
        self._label_transform = None
        self._competing = False
        self._n_risks = 1
        self._duration_index = None

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "DeepHitModel":
        import torch
        import torchtuples as tt
        from pycox.preprocessing.label_transforms import LabTransDiscreteTime

        np.random.seed(kwargs.get("seed", 42))
        torch.manual_seed(kwargs.get("seed", 42))

        # Detect competing risks: events > 1 means multiple causes
        unique_events = set(int(e) for e in E if e > 0)
        self._competing = len(unique_events) > 1
        self._n_risks = len(unique_events) if self._competing else 1

        # Discretise time
        self._label_transform = LabTransDiscreteTime(self._num_durations)
        self._label_transform.fit(T, E.astype("int64"))
        self._duration_index = self._label_transform.cuts

        # Transform labels
        y_disc = self._label_transform.transform(T, E.astype("int64"))
        # y_disc = (idx_durations, events) both int64

        # Build network
        in_features = X.shape[1]
        if self._competing:
            out_features = self._n_risks * len(self._duration_index)
        else:
            out_features = len(self._duration_index)

        net = tt.practical.MLPVanilla(
            in_features,
            self._hidden_layers,
            out_features,
            batch_norm=True,
            dropout=self._dropout,
        )

        # Build model
        if self._competing:
            from pycox.models import DeepHit
            self._model = DeepHit(
                net,
                tt.optim.Adam(self._lr),
                alpha=self._alpha,
                sigma=self._sigma,
                duration_index=self._duration_index,
            )
        else:
            from pycox.models import DeepHitSingle
            self._model = DeepHitSingle(
                net,
                tt.optim.Adam(self._lr),
                duration_index=self._duration_index,
                alpha=self._alpha,
                sigma=self._sigma,
            )

        # Train / val split
        x = X.astype("float32")
        idx_dur = y_disc[0].astype("int64")
        events = y_disc[1].astype("int64")

        n = len(X)
        n_val = int(n * self._val_fraction)
        perm = np.random.permutation(n)
        idx_v, idx_t = perm[:n_val], perm[n_val:]

        x_train, x_val = x[idx_t], x[idx_v]
        y_train = (idx_dur[idx_t], events[idx_t])
        y_val = (idx_dur[idx_v], events[idx_v])

        callbacks = [tt.callbacks.EarlyStopping(patience=self._patience)]
        self._model.fit(
            x_train, y_train,
            batch_size=self._batch_size,
            epochs=self._epochs,
            callbacks=callbacks,
            val_data=(x_val, y_val),
            verbose=False,
        )
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Risk score: negative mean predicted survival (higher = worse)."""
        x = X.astype("float32")
        surv_df = self._model.predict_surv_df(x)
        # Mean survival across time as a proxy; negate so higher = worse
        return -surv_df.values.mean(axis=0)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        x = X.astype("float32")
        surv_df = self._model.predict_surv_df(x)
        return self._interpolate_surv(surv_df, times)

    def predict_cumulative_incidence(
        self,
        X: np.ndarray,
        times: np.ndarray,
        cause: int,
    ) -> np.ndarray:
        """Predict cause-specific CIF for the competing risks variant.

        Parameters
        ----------
        cause : 1-indexed cause (matching the event encoding).
        """
        if not self._competing:
            raise NotImplementedError(
                "CIF prediction requires competing risks mode. "
                "Fit with multi-cause event indicators."
            )

        x = X.astype("float32")
        # predict_cif returns tensor of shape (n_risks, n_durations, n_subjects)
        cif_tensor = self._model.predict_cif(x)
        if hasattr(cif_tensor, "numpy"):
            cif_array = cif_tensor.numpy()
        else:
            cif_array = np.array(cif_tensor)

        # cause is 1-indexed in data, 0-indexed in the cif array
        cause_idx = cause - 1
        # cif_array[cause_idx] has shape (n_durations, n_subjects)
        cif_for_cause = cif_array[cause_idx]  # (n_durations, n_subjects)

        # Interpolate to requested times
        index = np.array(self._duration_index, dtype=float)
        n_subjects = cif_for_cause.shape[1]
        out = np.zeros((n_subjects, len(times)))
        for j in range(n_subjects):
            out[j] = np.interp(times, index, cif_for_cause[:, j], left=0.0, right=1.0)
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _interpolate_surv(surv_df, times: np.ndarray) -> np.ndarray:
        index = surv_df.index.values.astype(float)
        values = surv_df.values  # (n_time_grid, n_subjects)
        n_subjects = values.shape[1]
        out = np.zeros((n_subjects, len(times)))
        for j in range(n_subjects):
            out[j] = np.interp(times, index, values[:, j], left=1.0, right=0.0)
        return np.clip(out, 0.0, 1.0)
