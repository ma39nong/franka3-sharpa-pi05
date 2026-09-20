"""Compatibility import for the shared bridge runtime."""
import sys as _sys
from deploy.fr3_wuji_runtime import bridge as _implementation
_sys.modules[__name__] = _implementation
if __name__ == "__main__":
    _implementation.main()
