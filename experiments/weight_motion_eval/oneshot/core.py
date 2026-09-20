"""Compatibility import for the relocated FR3/Wuji implementation."""
import sys as _sys
from deploy.fr3_wuji_slow import core as _implementation
_sys.modules[__name__] = _implementation
