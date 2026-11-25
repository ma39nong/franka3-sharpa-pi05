"""Idle Home pedal, controller serialization and arm-only ROS playback."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_control import controller as control
from deploy.fr3_sharpa_control.ui import HTML
from deploy.fr3_sharpa_runtime import home_arms
from deploy.fr3_sharpa_runtime.protocol import FR3_COMMAND_NAMES


@pytest.fixture
def home_repository(tmp_path):
    root = tmp_path / 'deployment'
    launcher = root / 'deploy/fr3_sharpa_runtime/home_arms.sh'
    launcher.parent.mkdir(parents=True)
    launcher.touch()
    poses = root / '.deployment/operator_poses.yaml'
    poses.parent.mkdir()
    values = [-.1, -.5, 0., -1.8, 0., 1.5, 0.]
    poses.write_text(json.dumps({'tasks': {'powder_weighing': {'home': {
        side: {'joint_names': [f'{side}_fr3_joint{i}' for i in range(1, 8)], 'positions': values}
        for side in ('left', 'right')
    }}}}))
    return root


def test_idle_home_runs_without_checkpoint_and_serializes_execution(home_repository, monkeypatch):
    controller = control.DeploymentController(home_repository)
    started, release = threading.Event(), threading.Event()
    calls = []

    class Process:
        returncode = None
        def wait(self):
            started.set()
            assert release.wait(3)
            self.returncode = 0

    def spawn(command, **kwargs):
        calls.append(command)
        return Process()

    monkeypatch.setattr(control.subprocess, 'Popen', spawn)
    try:
        result = controller.home_arms(str(home_repository))
        assert result['accepted'] and started.wait(2)
        assert controller.home_arms(str(home_repository))['ignored']
        with pytest.raises(ValueError, match='等待双臂回Home'):
            controller.start({})
        assert len(calls) == 1 and '--home-poses' in calls[0]
    finally:
        release.set()
        controller.home_thread.join(3)
    assert controller.snapshot()['home']['phase'] == 'finished'


def test_home_is_ignored_during_execution_before_reading_configuration(tmp_path):
    controller = control.DeploymentController(tmp_path)
    controller.operation = 'execute'
    controller.state.update(running=True)
    assert controller.home_arms('/missing/repository')['ignored']
    assert controller.home_thread is None


def test_home_http_endpoint_requires_token_and_routes_repository(home_repository, monkeypatch):
    from deploy.fr3_sharpa_control.ui import server_for
    server = server_for(home_repository, port=0)
    calls = []
    monkeypatch.setattr(server.controller, 'home_arms', lambda repo: (calls.append(repo) or {'accepted': True}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        info = json.load(urllib.request.urlopen(url+'/api/info'))
        request = urllib.request.Request(url+'/api/home-arms', data=json.dumps(
            {'repository': str(home_repository)}).encode(), headers={'Content-Type': 'application/json'})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403 and not calls
        request.add_header('X-Control-Token', info['token'])
        assert json.load(urllib.request.urlopen(request))['accepted']
        assert calls == [str(home_repository)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize('cancel', [False, True])
def test_cleanup_accepts_one_home_request_and_waits_for_release(home_repository, monkeypatch, cancel):
    controller = control.DeploymentController(home_repository)
    cleanup_release = threading.Event()
    started = threading.Event()
    controller.operation = 'execute'
    controller.state.update(running=True)
    controller.cleanup_running = True

    def cleanup():
        assert cleanup_release.wait(3)
        controller.update('stopped', 'cleanup complete', running=False)

    controller.thread = threading.Thread(target=cleanup)
    controller.thread.start()
    monkeypatch.setattr(control.subprocess, 'Popen', lambda *args, **kwargs:
        (started.set() or SimpleNamespace(wait=lambda: None, returncode=0)))
    try:
        assert controller.home_arms(str(home_repository))['home']['phase'] == 'queued'
        assert not started.is_set()
        assert controller.home_arms(str(home_repository))['ignored']
        if cancel:
            controller.stop()
        cleanup_release.set()
        controller.home_thread.join(3)
        assert not controller.home_thread.is_alive()
        assert started.is_set() is not cancel
        assert controller.snapshot()['home']['phase'] == ('stopped' if cancel else 'finished')
    finally:
        cleanup_release.set()
        controller.thread.join(3)


def test_home_pedal_is_idle_only_and_cleanup_queues_without_key_repeat():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for browser pedal tests')
    handler = HTML.split('function homePedalKey(event){', 1)[1].split('async function requestArmsHome()', 1)[0]
    script = """
const assert=require('node:assert/strict');
let homePedalDown=false,homeRequestPending=false,homeRunning=false,executionCleaning=false;
let executionActive=false,executionStarting=false,editing=false,dialog=false,calls=0;
const document={querySelector:()=>dialog};
function requestArmsHome(){if(!executionActive||executionCleaning)calls++}
""" + 'function homePedalKey(event){' + handler + """
function key(extra={}){return {code:'KeyB',type:'keydown',repeat:false,
 target:{closest:()=>editing},preventDefault(){},...extra}}
function tap(extra={}){homePedalKey(key(extra));homePedalKey(key({type:'keyup'}))}
homePedalKey(key());homePedalKey(key());homePedalKey(key({repeat:true}));
assert.equal(calls,1);homePedalKey(key({type:'keyup'}));
executionActive=true;tap();assert.equal(calls,1);
executionCleaning=true;tap();assert.equal(calls,2);
homeRunning=true;tap();homeRunning=false;
homeRequestPending=true;tap();homeRequestPending=false;
executionActive=false;executionCleaning=false;
executionStarting=true;tap();executionStarting=false;
editing=true;tap();editing=false;dialog=true;tap();dialog=false;
tap({target:{isContentEditable:true}});
for(const modifier of ['ctrlKey','altKey','metaKey','shiftKey','isComposing'])tap({[modifier]:true});
tap({code:'KeyA'});assert.equal(calls,2);
tap();assert.equal(calls,3);
"""
    result = subprocess.run([node, '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_arm_home_commands_are_bounded_and_reject_tracking_faults():
    start = np.zeros(14)
    target = np.ones(14)
    lower, upper = np.full(14, -2.), np.full(14, 2.)
    command = home_arms.home_command(start, target, start, lower, upper)
    np.testing.assert_allclose(command, .1/30, atol=1e-9)
    with pytest.raises(ValueError, match='tracking error'):
        home_arms.home_command(np.ones(14), target, start, lower, upper)
    with pytest.raises(ValueError, match='joint limits'):
        home_arms.home_command(start, target*3, start, lower, upper)


@pytest.mark.parametrize('scenario', ['success', 'foreign', 'startup-failure'])
def test_home_session_starts_only_arm_services_and_releases_owned_hardware(home_repository, monkeypatch, scenario):
    from deploy.fr3_sharpa_runtime import camera_lease, hardware_session
    gello = home_repository/'hardware'
    (gello/'docker').mkdir(parents=True)
    (gello/'docker/.env').write_text('TELEOP_ROS_DOMAIN_ID=31\n')
    (gello/'portable_deps/litchi_hardware').mkdir(parents=True)
    overlay = home_repository/'overlay'
    overlay.mkdir()
    (overlay/'setup.bash').touch()
    monkeypatch.setenv('SHARPA_GELLO_REPO', str(gello))
    monkeypatch.setenv('SHARPA_ROS_MESSAGES_OVERLAY', str(overlay))
    calls = []

    class Lease:
        def acquire(self):
            calls.append('lease')
            return self
        def close(self):
            calls.append('release')

    class Hardware:
        compose = ['docker', 'compose']
        def __init__(self):
            self.owned = {'teleop-control': {'State': {'Running': True}}} if scenario == 'foreign' else {}
        def is_openpi(self, item, service):
            return False
        def inventory(self):
            return {'teleop-control': {}, 'franka-control': {}, 'openpi-cameras': {}}
        def cleanup(self):
            assert 'openpi-cameras' not in self.owned
            calls.append('cleanup')

    def compose(command, **kwargs):
        calls.append(command)
        assert command[-2:] == ['franka-control', 'teleop-control']
        if scenario == 'startup-failure':
            raise subprocess.CalledProcessError(1, command)

    def session(command, hardware):
        assert command[:5] == ['pixi', 'run', '--locked', '-e', 'ros2']
        assert '--ros' in command[-1] and '--home-poses' in command[-1]
        hardware.cleanup()
        return 0

    monkeypatch.setattr(camera_lease, 'CameraLease', Lease)
    monkeypatch.setattr(hardware_session, 'HardwareSession', Hardware)
    monkeypatch.setattr(hardware_session, 'run_session', session)
    monkeypatch.setattr(home_arms.subprocess, 'run', compose)
    poses = home_repository/'.deployment/operator_poses.yaml'
    if scenario == 'foreign':
        with pytest.raises(RuntimeError, match='owns the gateway'):
            home_arms.run_home(poses, 'powder_weighing')
        assert calls == ['lease', 'release']
    elif scenario == 'startup-failure':
        with pytest.raises(subprocess.CalledProcessError):
            home_arms.run_home(poses, 'powder_weighing')
        assert calls[-2:] == ['cleanup', 'release']
    else:
        assert home_arms.run_home(poses, 'powder_weighing') == 0
        assert calls[-2:] == ['cleanup', 'release']


@pytest.mark.parametrize('stale', [False, True])
def test_ros_home_uses_only_arm_messages_and_requires_fresh_feedback(home_repository, monkeypatch, stale):
    from deploy.fr3_sharpa_runtime import ros_transport
    from deploy.fr3_sharpa_runtime.ros_observation import TOPICS
    clock = [1.]
    state = np.array([-.08, -.5, 0., -1.8, 0., 1.5, 0.]*2)
    commands, closed = [], []
    callbacks = {}

    class ArmCommand:
        def __init__(self):
            self.header = SimpleNamespace()

    class Node:
        def create_subscription(self, kind, topic, callback, depth):
            callbacks[topic] = callback
        def get_clock(self):
            return SimpleNamespace(now=lambda: SimpleNamespace(
                nanoseconds=int(clock[0]*1e9), to_msg=lambda: int(clock[0]*1e9)))
        def get_node_names(self):
            return ['teleop_safety_gateway']
        def get_publishers_info_by_topic(self, topic):
            return []
        def create_publisher(self, kind, topic, depth):
            assert kind is ArmCommand and topic == '/teleop/arm_commands'
            def publish(message):
                assert message.joint_names == list(FR3_COMMAND_NAMES)
                command = np.array(message.positions)
                commands.append(command)
                state[:] = command
                callbacks['/teleop/arm_command_status'](SimpleNamespace(
                    session_id=message.session_id, source='openpi', sequence=message.sequence,
                    accepted_sides=['left', 'right'], faults=[]))
            return SimpleNamespace(publish=publish)
        def destroy_node(self):
            closed.append('node')

    def spin_once(node, timeout_sec):
        clock[0] += .001
        for i, side in enumerate(('left', 'right')):
            if stale and clock[0] > 1.1:
                continue
            stamp = int(clock[0]*1e9)
            callbacks[TOPICS[side+'_arm']](SimpleNamespace(
                name=list(FR3_COMMAND_NAMES[i*7:i*7+7]), position=state[i*7:i*7+7].tolist(),
                header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp//10**9, nanosec=stamp%10**9))))

    monkeypatch.setitem(sys.modules, 'rclpy', SimpleNamespace(init=lambda: None, create_node=lambda name: Node(),
        spin_once=spin_once, ok=lambda: True, shutdown=lambda: closed.append('ros')))
    monkeypatch.setitem(sys.modules, 'sensor_msgs.msg', SimpleNamespace(JointState=object))
    monkeypatch.setitem(sys.modules, 'teleop_interfaces.msg', SimpleNamespace(ArmCommand=ArmCommand, ArmCommandStatus=object))
    monkeypatch.setitem(sys.modules, 'teleop_core.safety', SimpleNamespace(
        LOWER_LIMITS=np.full(14, -3.), UPPER_LIMITS=np.full(14, 3.)))
    monkeypatch.setattr(ros_transport, 'validate_gateway_admission', lambda node: None)
    monkeypatch.setattr(ros_transport, 'read_gateway_joint_speed', lambda node: .7)
    monkeypatch.setattr(home_arms, 'time', SimpleNamespace(monotonic=lambda: clock[0],
        monotonic_ns=lambda: int(clock[0]*1e9)))
    poses = home_repository/'.deployment/operator_poses.yaml'
    if stale:
        with pytest.raises(ValueError, match='Stale/future arm feedback'):
            home_arms.ros_home(poses, 'powder_weighing')
    else:
        home_arms.ros_home(poses, 'powder_weighing')
        np.testing.assert_allclose(commands[-1], [-.1, -.5, 0., -1.8, 0., 1.5, 0.]*2, atol=1e-6)
    assert commands and closed == ['node', 'ros']
    assert set(callbacks) == {TOPICS['left_arm'], TOPICS['right_arm'], '/teleop/arm_command_status'}
