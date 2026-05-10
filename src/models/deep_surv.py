"""DeepSurv wrapper around pycox CoxPH model with neural network."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# NumPy 2.x removed several aliases that numba/pycox still reference — patch them.
for _old, _new in {"trapz": "trapezoid", "in1d": "isin", "row_stack": "vstack",
                    "product": "prod", "cumproduct": "cumprod",
                    "sometrue": "any", "alltrue": "all"}.items():
    if not hasattr(np, _old) and hasattr(np, _new):
        setattr(np, _old, getattr(np, _new))

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
        weight_decay: float = 0.0,
        epochs: int = 100,
        batch_size: int = 256,
        val_fraction: float = 0.2,
        patience: int = 10,
    ) -> None:
        self._hidden_layers = hidden_layers or [64, 64]
        self._dropout = dropout
        self._lr = lr
        self._weight_decay = weight_decay
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

        # Move to GPU if available
        if torch.cuda.is_available():
            net.cuda()

        self._model = CoxPH(net, tt.optim.Adam(self._lr, weight_decay=self._weight_decay))

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
        self, X: np.ndarray, times: np.ndarray,
    ) -> np.ndarray:
        """Predict S(t|X) = exp(-H0(t) * exp(log_ph(x)))."""
        log_ph = self._model.predict(X.astype("float32")).flatten()  # (n,)

        bh = self._model.baseline_hazards_.copy()
        bh.index = bh.index.astype(float)
        bh = bh.sort_index().groupby(level=0).sum()

        event_times = bh.index.values.astype(float)
        increments = np.clip(bh.values.flatten().astype(float), 0.0, None)
        cum_bh = np.maximum.accumulate(np.cumsum(increments))
        H0_at_times = np.interp(
            times,
            event_times,
            cum_bh,
            left=0.0,
            right=cum_bh[-1],
        )

        out = np.exp(-H0_at_times[None, :] * np.exp(log_ph)[:, None])
        return np.clip(out, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        import torch, pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "net_state_dict": self._model.net.state_dict(),
            "baseline_hazards": self._model.compute_baseline_hazards(),
            "params": {
                "hidden_layers": self._hidden_layers,
                "dropout": self._dropout,
                "lr": self._lr,
                "weight_decay": self._weight_decay,
                "epochs": self._epochs,
                "batch_size": self._batch_size,
                "val_fraction": self._val_fraction,
                "patience": self._patience,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "DeepSurvModel":
        import torch, pickle
        import torchtuples as tt
        from pycox.models import CoxPH

        with open(path, "rb") as f:
            state = pickle.load(f)
        params = state["params"]
        obj = cls(**params)
        # We need in_features to rebuild the net — infer from state_dict
        first_key = next(iter(state["net_state_dict"]))
        in_features = state["net_state_dict"][first_key].shape[1]
        net = tt.practical.MLPVanilla(
            in_features, params["hidden_layers"], 1,
            batch_norm=True, dropout=params["dropout"],
        )
        obj._model = CoxPH(net, tt.optim.Adam(params["lr"]))
        net.load_state_dict(state["net_state_dict"])
        if torch.cuda.is_available():
            net.cuda()
        obj._model.baseline_hazards_ = state["baseline_hazards"]
        obj._fitted = True
        return obj

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
