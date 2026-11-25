import json
import re
import shutil
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from deploy.fr3_sharpa_control.controller import DeploymentController
from deploy.fr3_sharpa_control.ui import HTML, server_for


@pytest.mark.parametrize('mode,directory', [('manual', ''), ('auto', '/data/采集 目录')])
def test_manual_preferences_persist_immediately_and_survive_restart(tmp_path, mode, directory):
    controller = DeploymentController(tmp_path)
    controller.save_ui_preferences({'stage_switch_mode': mode})
    controller.save_ui_preferences({'dagger_output': directory})
    restored = DeploymentController(tmp_path)
    assert restored.ui_preferences == {'stage_switch_mode': mode, 'dagger_output': directory}
    assert restored.thread is None and restored.runtime is None
    assert restored.agent.mode == 'manual'
    assert not (tmp_path/'.deployment/control_last.json').exists()


def test_concurrent_partial_preference_updates_preserve_both_choices(tmp_path):
    controller = DeploymentController(tmp_path)
    other = DeploymentController(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(controller.save_ui_preferences, {'stage_switch_mode': 'auto'}),
                   pool.submit(other.save_ui_preferences, {'dagger_output': '/data/dagger'})]
        for future in futures:
            future.result()
    assert DeploymentController(tmp_path).ui_preferences == {
        'stage_switch_mode': 'auto', 'dagger_output': '/data/dagger'}


@pytest.mark.parametrize('invalid', [{'stage_switch_mode': 'unknown'}, {'dagger_output': False}, {'checkpoint': 'other'}])
def test_invalid_preference_update_keeps_saved_choices(tmp_path, invalid):
    controller = DeploymentController(tmp_path)
    expected = controller.save_ui_preferences({'stage_switch_mode': 'manual'})
    with pytest.raises(ValueError):
        controller.save_ui_preferences(invalid)
    assert controller.ui_preferences == expected == DeploymentController(tmp_path).ui_preferences


def test_preferences_endpoint_and_info_reload_saved_values(tmp_path):
    def start():
        server = server_for(tmp_path, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, f'http://127.0.0.1:{server.server_port}'

    server, thread, url = start()
    expected = {'stage_switch_mode': 'auto', 'dagger_output': '/data/new dagger'}
    try:
        info = json.load(urllib.request.urlopen(url+'/api/info'))
        request = urllib.request.Request(url+'/api/preferences', data=json.dumps(expected).encode(),
            headers={'Content-Type': 'application/json', 'X-Control-Token': info['token']})
        assert json.load(urllib.request.urlopen(request)) == expected
        assert json.load(urllib.request.urlopen(url+'/api/info'))['ui_preferences'] == expected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
    server, thread, url = start()
    try:
        assert json.load(urllib.request.urlopen(url+'/api/info'))['ui_preferences'] == expected
        assert server.controller.thread is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_browser_saves_mode_typed_and_browsed_paths_and_restores_without_activation():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser preference handlers')
    persistence = HTML.split('let preferenceSaves=Promise.resolve();', 1)[1].split('function invalidate()', 1)[0]
    mode_handler = HTML.split('function agentModeChanged(){', 1)[1].split('let taskState=', 1)[0]
    choose_path = HTML.split('function choosePath(path){', 1)[1].split('\n', 1)[0]
    input_handler = re.search(r'<input id="dagger_output" oninput="([^"]+)"', HTML)[1]
    harness = """
const assert=require('node:assert/strict');
const nodes=new Map(), calls=[];
const $=id=>{if(!nodes.has(id))nodes.set(id,{value:'',textContent:'',close(){}});return nodes.get(id)};
let autoModePending=false,agentModeRequest=null,agentState={mode:'manual'},pickerTarget='dagger_output';
let preferenceSaves=Promise.resolve();
async function api(path,values){assert.equal(path,'preferences');calls.push(values);return values}
function failure(error){throw error}
function updateAgentModeControl(){$('agentMode').value=autoModePending?'auto':agentState.mode}
function applyPendingAgentMode(){}
function setAgentMode(mode){agentState.mode=mode}
function invalidate(){}
function repositoryChanged(){}
function modelTypeChanged(){}
"""
    scenario = """
(async()=>{
 restoreControlPreferences({stage_switch_mode:'auto',dagger_output:'/data/restored'});
 assert.equal($('agentMode').value,'auto');assert.equal($('dagger_output').value,'/data/restored');
 assert.equal(agentState.mode,'manual');assert.deepEqual(calls,[]);
 $('agentMode').value='auto';agentModeChanged();
 $('dagger_output').value='/data/typed';
 TYPED_HANDLER
 $('agentMode').value='manual';agentModeChanged();
 choosePath('/data/browsed');
 await preferenceSaves;
 assert.deepEqual(calls,[{stage_switch_mode:'auto'},{dagger_output:'/data/typed'},
                         {stage_switch_mode:'manual'},{dagger_output:'/data/browsed'}]);
 assert.equal($('agentMode').value,'manual');
})().catch(error=>{console.error(error);process.exitCode=1});
""".replace('TYPED_HANDLER', '(function(){'+input_handler+'}).call($("dagger_output"));')
    script = harness + persistence + 'function agentModeChanged(){' + mode_handler
    script += 'function choosePath(path){' + choose_path + '\n' + scenario
    result = subprocess.run([node, '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
