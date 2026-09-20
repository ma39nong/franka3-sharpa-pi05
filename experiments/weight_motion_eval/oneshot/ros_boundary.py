"""Compatibility import for the shared ros_boundary runtime."""
import sys as _sys
from deploy.fr3_wuji_runtime import ros_boundary as _implementation
_sys.modules[__name__] = _implementation
if __name__ == "__main__":
    _implementation.main()
