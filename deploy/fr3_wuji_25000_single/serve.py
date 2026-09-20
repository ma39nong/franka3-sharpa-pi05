"""Compatibility import for the model adapter in deploy.fr3_wuji_models."""
import sys as _sys
from deploy.fr3_wuji_models.model_25000_single import serve as _implementation
_sys.modules[__name__] = _implementation
if __name__ == "__main__":
    _implementation.main()
