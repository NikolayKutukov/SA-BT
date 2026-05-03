"""Survival model wrappers with unified interface."""

from .base import SurvivalModel
from .cox_ph import CoxPHModel
from .cox_net import CoxNetModel
from .rsf import RSFModel
from .deep_surv import DeepSurvModel

ALL_MODELS = {
    "cox_ph": CoxPHModel,
    "cox_net": CoxNetModel,
    "rsf": RSFModel,
    "deep_surv": DeepSurvModel,
}

__all__ = [
    "SurvivalModel",
    "CoxPHModel",
    "CoxNetModel",
    "RSFModel",
    "DeepSurvModel",
    "ALL_MODELS",
]
