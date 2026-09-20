"""Compatibility import for the relocated FR3/Wuji implementation."""
import sys as _sys
from deploy.fr3_wuji_slow import deploy as _implementation
if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
