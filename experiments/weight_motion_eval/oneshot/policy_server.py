"""Compatibility import for the 19999 model service."""
import sys as _sys
from deploy.fr3_wuji_models.model_19999 import serve as _implementation
_sys.modules[__name__] = _implementation
if __name__ == "__main__":
    _implementation.main()
