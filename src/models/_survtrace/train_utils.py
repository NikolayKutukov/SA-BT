"""Training utilities for SurvTRACE (vendored + cleaned up).

Adapted from https://github.com/RyanWangZf/SurvTRACE (MIT License).
Fixed deprecated PyTorch API calls for torch 2.x compatibility.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from .losses import NLLPCHazardLoss


class EarlyStopping:
    """Early stops training if validation loss doesn't improve."""

    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss: float, model, name: str = "checkpoint.pt"):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model, name)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model, name):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), name)
        self.val_loss_min = val_loss


# ── Learning-rate schedules ──────────────────────────────────────────

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + math.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


SCHEDULES = {
    "warmup_cosine": warmup_cosine,
    "warmup_constant": warmup_constant,
    "warmup_linear": warmup_linear,
}


# ── BERTAdam optimizer ───────────────────────────────────────────────

class BERTAdam(Optimizer):
    """BERT version of Adam with weight-decay fix."""

    def __init__(
        self,
        params,
        lr: float,
        warmup: float = -1,
        t_total: int = -1,
        schedule: str = "warmup_linear",
        b1: float = 0.9,
        b2: float = 0.999,
        e: float = 1e-6,
        weight_decay_rate: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        defaults = dict(
            lr=lr,
            schedule=schedule,
            warmup=warmup,
            t_total=t_total,
            b1=b1,
            b2=b2,
            e=e,
            weight_decay_rate=weight_decay_rate,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["next_m"] = torch.zeros_like(p.data)
                    state["next_v"] = torch.zeros_like(p.data)

                next_m, next_v = state["next_m"], state["next_v"]
                beta1, beta2 = group["b1"], group["b2"]

                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])

                # Decay first and second moment estimates
                next_m.mul_(beta1).add_(grad, alpha=1 - beta1)
                next_v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = next_m / (next_v.sqrt() + group["e"])

                if group["weight_decay_rate"] > 0.0:
                    update += group["weight_decay_rate"] * p.data

                if group["t_total"] != -1:
                    schedule_fct = SCHEDULES[group["schedule"]]
                    lr_scheduled = group["lr"] * schedule_fct(
                        state["step"] / group["t_total"], group["warmup"]
                    )
                else:
                    lr_scheduled = group["lr"]

                p.data.add_(update, alpha=-lr_scheduled)
                state["step"] += 1

        return loss


# ── Trainer ──────────────────────────────────────────────────────────

class Trainer:
    """Orchestrates SurvTRACE training for single- and multi-event models."""

    def __init__(self, model, metrics=None):
        self.model = model
        self.metrics = metrics or [NLLPCHazardLoss()]
        self.train_logs = defaultdict(list)
        self.get_target = lambda df: (df["duration"].values, df["event"].values)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            self.model.use_gpu = True
        self.early_stopping = None

        ckpt_path = model.config.get("checkpoint", "./checkpoints/survtrace.pt")
        self.ckpt = ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        if ckpt_dir and not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

    def fit(
        self,
        train_set,
        val_set=None,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=0,
        val_batch_size=None,
        verbose=True,
        **kwargs,
    ):
        if self.model.config["num_event"] == 1:
            return self._train_single_event(
                train_set, val_set, batch_size, epochs,
                learning_rate, weight_decay, val_batch_size, verbose,
            )
        elif self.model.config["num_event"] > 1:
            return self._train_multi_event(
                train_set, val_set, batch_size, epochs,
                learning_rate, weight_decay, val_batch_size, verbose,
            )
        else:
            raise ValueError("num_event must be >= 1")

    def _make_optimizer(self, weight_decay, learning_rate):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        param_optimizer = list(self.model.named_parameters())
        grouped = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return BERTAdam(grouped, learning_rate, weight_decay_rate=weight_decay)

    def _train_single_event(
        self, train_set, val_set, batch_size, epochs,
        learning_rate, weight_decay, val_batch_size, verbose,
    ):
        df_train, df_y_train = train_set
        tensor_val = tensor_y_val = None

        if val_set is not None:
            tensor_val = torch.tensor(val_set[0].values)
            tensor_y_val = torch.tensor(val_set[1].values)
            if self.use_gpu:
                tensor_val = tensor_val.cuda()
                tensor_y_val = tensor_y_val.cuda()

        optimizer = self._make_optimizer(weight_decay, learning_rate)

        if val_set is not None:
            self.early_stopping = EarlyStopping(
                patience=self.model.config.get("early_stop_patience", 5)
            )

        num_train_batch = int(np.ceil(len(df_y_train) / batch_size))
        train_loss_list, val_loss_list = [], []

        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()

            df_train_shuffled = train_set[0].sample(frac=1)
            df_y_shuffled = train_set[1].loc[df_train_shuffled.index]

            tensor_train = torch.tensor(df_train_shuffled.values)
            tensor_y_train = torch.tensor(df_y_shuffled.values)
            if self.use_gpu:
                tensor_train = tensor_train.cuda()
                tensor_y_train = tensor_y_train.cuda()

            for batch_idx in range(num_train_batch):
                optimizer.zero_grad()
                batch_train = tensor_train[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_y = tensor_y_train[batch_idx * batch_size : (batch_idx + 1) * batch_size]

                batch_x_cat = batch_train[:, : self.model.config["num_categorical_feature"]].long()
                batch_x_num = batch_train[:, self.model.config["num_categorical_feature"] :].float()

                phi = self.model(input_ids=batch_x_cat, input_nums=batch_x_num)
                batch_loss = self.metrics[0](
                    phi[1], batch_y[:, 0].long(), batch_y[:, 1].long(), batch_y[:, 2].float()
                )
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

            train_loss_list.append(epoch_loss / (batch_idx + 1))

            if val_set is not None:
                self.model.eval()
                with torch.no_grad():
                    phi_val = self.model.predict(tensor_val, val_batch_size)
                val_loss = self.metrics[0](
                    phi_val, tensor_y_val[:, 0].long(), tensor_y_val[:, 1].long(), tensor_y_val[:, 2].float()
                )
                val_loss_list.append(val_loss.item())
                if verbose:
                    print(f"[Train-{epoch}]: {epoch_loss / (batch_idx + 1):.4f}")
                    print(f"[Val-{epoch}]: {val_loss.item():.4f}")
                self.early_stopping(val_loss.item(), self.model, name=self.ckpt)
                if self.early_stopping.early_stop:
                    if verbose:
                        print(f"Early stop at epoch {epoch + 1}")
                    self.model.load_state_dict(
                        torch.load(self.ckpt, weights_only=True)
                    )
                    return train_loss_list, val_loss_list
            else:
                if verbose:
                    print(f"[Train-{epoch}]: {epoch_loss / (batch_idx + 1):.4f}")

        return train_loss_list, val_loss_list

    def _train_multi_event(
        self, train_set, val_set, batch_size, epochs,
        learning_rate, weight_decay, val_batch_size, verbose,
    ):
        tensor_val = None
        tensor_y_val = {}
        num_event = self.model.config["num_event"]

        if val_set is not None:
            tensor_val = torch.tensor(val_set[0].values)
            for risk in range(num_event):
                cols = ["duration", f"event_{risk}", "proportion"]
                tensor_y_val[f"risk_{risk}"] = torch.tensor(val_set[1][cols].values)
            if self.use_gpu:
                tensor_val = tensor_val.cuda()
                for k in tensor_y_val:
                    tensor_y_val[k] = tensor_y_val[k].cuda()

        optimizer = self._make_optimizer(weight_decay, learning_rate)

        if val_set is not None:
            self.early_stopping = EarlyStopping(
                patience=self.model.config.get("early_stop_patience", 5)
            )

        num_train_batch = int(np.ceil(len(train_set[0]) / batch_size))
        train_loss_list, val_loss_list = [], []

        for epoch in range(epochs):
            df_train_shuffled = train_set[0].sample(frac=1)
            df_y_shuffled = train_set[1].loc[df_train_shuffled.index]

            tensor_train = torch.tensor(df_train_shuffled.values)
            tensor_y_train = {}
            for risk in range(num_event):
                cols = ["duration", f"event_{risk}", "proportion"]
                tensor_y_train[f"risk_{risk}"] = torch.tensor(df_y_shuffled[cols].values)

            if self.use_gpu:
                tensor_train = tensor_train.cuda()
                for k in tensor_y_train:
                    tensor_y_train[k] = tensor_y_train[k].cuda()

            epoch_loss = 0
            self.model.train()
            for batch_idx in range(num_train_batch):
                optimizer.zero_grad()
                batch_train = tensor_train[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_x_cat = batch_train[:, : self.model.config["num_categorical_feature"]].long()
                batch_x_num = batch_train[:, self.model.config["num_categorical_feature"] :].float()

                batch_loss = None
                for risk in range(num_event):
                    phi = self.model(input_ids=batch_x_cat, input_nums=batch_x_num, event=risk)
                    batch_y = tensor_y_train[f"risk_{risk}"][
                        batch_idx * batch_size : (batch_idx + 1) * batch_size
                    ]
                    loss_k = self.metrics[0](
                        phi[1], batch_y[:, 0].long(), batch_y[:, 1].long(), batch_y[:, 2].float()
                    )
                    batch_loss = loss_k if batch_loss is None else batch_loss + loss_k

                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

            train_loss_list.append(epoch_loss / (batch_idx + 1))

            if val_set is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for risk in range(num_event):
                        phi_val = self.model.predict(tensor_val, val_batch_size, event=risk)
                        val_loss += self.metrics[0](
                            phi_val,
                            tensor_y_val[f"risk_{risk}"][:, 0].long(),
                            tensor_y_val[f"risk_{risk}"][:, 1].long(),
                            tensor_y_val[f"risk_{risk}"][:, 2].float(),
                        )
                val_loss_list.append(val_loss.item())
                if verbose:
                    print(f"[Train-{epoch}]: {epoch_loss / (batch_idx + 1):.4f}")
                    print(f"[Val-{epoch}]: {val_loss.item():.4f}")
                self.early_stopping(val_loss.item(), self.model, name=self.ckpt)
                if self.early_stopping.early_stop:
                    if verbose:
                        print(f"Early stop at epoch {epoch + 1}")
                    self.model.load_state_dict(
                        torch.load(self.ckpt, weights_only=True)
                    )
                    return train_loss_list, val_loss_list
            else:
                if verbose:
                    print(f"[Train-{epoch}]: {epoch_loss / (batch_idx + 1):.4f}")

        return train_loss_list, val_loss_list
