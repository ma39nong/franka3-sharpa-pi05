"""Fast profile adapter for the shared bounded IPC client."""

from deploy.fr3_wuji_runtime.ipc import DeviceClient
from deploy.fr3_wuji_slow.core import Feedback, checked_arm_speed
from .timeline import ARM_SPEED_RAD_S


class RemoteDevices(DeviceClient):
    feedback_type = Feedback

    def __init__(self, path, *, execute=False, finish_policy="hold", sock=None, arm_speed_rad_s=ARM_SPEED_RAD_S):
        super().__init__(
            path, feedback_type=Feedback, speed_validator=checked_arm_speed,
            execute=execute, finish_policy=finish_policy, sock=sock,
            arm_speed_rad_s=arm_speed_rad_s,
        )
