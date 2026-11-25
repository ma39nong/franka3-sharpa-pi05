"""Only GELLO + MANUS capture/retarget nodes. No arm controller or hand SDK."""
import os
from pathlib import Path
import sys
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    repo = Path(os.environ['SHARPA_GELLO_REPO'])
    hardware = repo/'portable_deps/litchi_hardware'
    values = {}
    for line in (repo/'docker/.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('"\'')
    native = values.get('MANUS_NATIVE_LIBRARY', '')
    calibration = values.get('MANUS_CALIBRATION_DIRECTORY', str(hardware/'calibration/metaglove'))
    if (native and not Path(native).is_file()) or not Path(calibration).is_dir():
        raise FileNotFoundError('Configure MANUS_NATIVE_LIBRARY / MANUS_CALIBRATION_DIRECTORY in gello docker/.env')
    driver = Node(package='manus_driver', executable='manus_driver', name='dagger_manus_input',
        parameters=[dict(use_fake_hardware=False, backend='native', native_library=native,
                         calibration_directory=calibration, publish_left=True, publish_right=False)])
    retarget = Node(package='manus_sharpa_teleop', executable='retarget_node', name='dagger_sharpa_retarget',
        parameters=[dict(enable_left=True, enable_right=False, backend='v4', max_input_age_s=.2,
                         max_output_age_s=.2, left_command_topic='/dagger/input/left_hand')],
        remappings=[('/sharpa/left/command','/dagger/input/left_hand')])
    gello = ExecuteProcess(cmd=[sys.executable, '-m', 'deploy.fr3_sharpa_runtime.dagger_inputs'], output='screen')
    processes = [driver, retarget, gello]
    return LaunchDescription([*processes, *[
        RegisterEventHandler(OnProcessExit(target_action=process, on_exit=[Shutdown(reason='DAgger input process exited')]))
        for process in processes]])
