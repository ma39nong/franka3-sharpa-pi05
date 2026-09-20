"""Speed-profile types for the shared bounded IPC client."""
from deploy.fr3_wuji_runtime.ipc import MAX_PACKET
from deploy.fr3_wuji_runtime.ipc import DeviceClient
from deploy.fr3_wuji_runtime.ipc import encode
from deploy.fr3_wuji_runtime.ipc import receive
from deploy.fr3_wuji_runtime.ipc import unpack_frame as _unpack_frame
from deploy.fr3_wuji_runtime.ipc import wire_feedback
from deploy.fr3_wuji_runtime.ipc import wire_frame
from .core import Feedback
from .core import Frame
from .core import checked_arm_speed


class RemoteDevices(DeviceClient):
    feedback_type = Feedback
    def __init__(self, path, *, execute=False, finish_policy="hold", sock=None, arm_speed_rad_s=2.0):
        super().__init__(
            path, feedback_type=Feedback, speed_validator=checked_arm_speed,
            execute=execute, finish_policy=finish_policy, sock=sock,
            arm_speed_rad_s=arm_speed_rad_s,
        )


def unpack_frame(value):
    return _unpack_frame(value, Frame)
