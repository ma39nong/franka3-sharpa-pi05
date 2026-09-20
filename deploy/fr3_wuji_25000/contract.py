"""Compatibility import for the model adapter in deploy.fr3_wuji_models."""
import sys as _sys
from deploy.fr3_wuji_models.model_25000 import contract as _implementation
_sys.modules[__name__] = _implementation
