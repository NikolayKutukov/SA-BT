"""DeepSurv wrapper around pycox CoxPH model with neural network."""

from __future__ import annotations

import numpy as np

from .base import SurvivalModel


class DeepSurvModel(SurvivalModel):
    """Cox PH with a deep neural network (DeepSurv).

    Wraps ``pycox.models.CoxPH`` with an MLP backbone.
    Preserves the PH assumption but learns nonlinear covariate effects.
    """

    name = "DeepSurv"

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        val_fraction: float = 0.2,
        patience: int = 10,
    ) -> None:
        self._hidden_layers = hidden_layers or [64, 64]
        self._dropout = dropout
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._val_fraction = val_fraction
        self._patience = patience
        self._model = None
        self._fitted = False

    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, **kwargs) -> "DeepSurvModel":
        import torch
        import torchtuples as tt
        from pycox.models import CoxPH

        np.random.seed(kwargs.get("seed", 42))
        torch.manual_seed(kwargs.get("seed", 42))

        in_features = X.shape[1]
        net = tt.practical.MLPVanilla(
            in_features,
            self._hidden_layers,
            1,
            batch_norm=True,
            dropout=self._dropout,
        )

        self._model = CoxPH(net, tt.optim.Adam(self._lr))

        # Train / val split
        x = X.astype("float32")
        y = (T.astype("float32"), E.astype("float32"))

        n = len(X)
        n_val = int(n * self._val_fraction)
        perm = np.random.permutation(n)
        idx_val, idx_train = perm[:n_val], perm[n_val:]

        x_train, x_val = x[idx_train], x[idx_val]
        y_train = (y[0][idx_train], y[1][idx_train])
        y_val = (y[0][idx_val], y[1][idx_val])

        callbacks = [tt.callbacks.EarlyStopping(patience=self._patience)]
        log = self._model.fit(
            x_train, y_train,
            batch_size=self._batch_size,
            epochs=self._epochs,
            callbacks=callbacks,
            val_data=(x_val, y_val),
            verbose=False,
        )

        # Compute baseline hazard from training data
        _ = self._model.compute_baseline_hazards()
        self._fitted = True
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        x = X.astype("float32")
        # CoxPH.predict returns log-partial hazard; higher = worse
        return self._model.predict(x).flatten()

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        x = X.astype("float32")
        surv_df = self._model.predict_surv_df(x)
        # surv_df: index = duration grid, columns = subjects
        # Interpolate to requested times
        return self._interpolate_surv(surv_df, times)

    @staticmethod
    def _interpolate_surv(surv_df, times: np.ndarray) -> np.ndarray:
        """Interpolate survival DataFrame at requested time points.

        Returns (n_subjects, len(times)) array.
        """
        index = surv_df.index.values.astype(float)
        values = surv_df.values  # (n_time_grid, n_subjects)
        n_subjects = values.shape[1]
        n_times = len(times)
        out = np.zeros((n_subjects, n_times))

        for j in range(n_subjects):
            out[j] = np.interp(times, index, values[:, j], left=1.0, right=0.0)

        return np.clip(out, 0.0, 1.0)
