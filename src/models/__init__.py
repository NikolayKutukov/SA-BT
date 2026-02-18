"""Survival model wrappers with unified interface."""

from .base import SurvivalModel, CauseSpecificWrapper
from .cox_ph import CoxPHModel
from .cox_net import CoxNetModel
from .rsf import RSFModel
from .deep_surv import DeepSurvModel
from .deep_hit import DeepHitModel
from .survtrace_wrapper import SurvTRACEModel

ALL_MODELS = {
    "cox_ph": CoxPHModel,
    "cox_net": CoxNetModel,
    "rsf": RSFModel,
    "deep_surv": DeepSurvModel,
    "deep_hit": DeepHitModel,
    "survtrace": SurvTRACEModel,
}

__all__ = [
    "SurvivalModel",
    "CauseSpecificWrapper",
    "CoxPHModel",
    "CoxNetModel",
    "RSFModel",
    "DeepSurvModel",
    "DeepHitModel",
    "SurvTRACEModel",
    "ALL_MODELS",
]
