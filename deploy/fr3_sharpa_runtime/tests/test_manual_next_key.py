import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_next_key_requires_active_manual_stage_and_ignores_typing_and_repeat():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for real UI key handler execution')
    source = (Path(__file__).resolve().parents[2] / 'fr3_sharpa_control/ui.py').read_text()
    handler = source.split('function manualNextKey(event){', 1)[1].split(
        "document.addEventListener('keydown',manualNextKey);", 1)[0]
    script = '''
const assert=require('node:assert/strict');
let taskState={execution_state:'RUNNING',requested_stage:'pickup',active_stage:'pickup'};
let taskPending=false,agentState={mode:'manual'},calls=0,editing=false,dialog=false,disabled=false;
const document={querySelector:()=>dialog};
const $=()=>({disabled});
function taskCommand(op){assert.equal(op,'next');calls++;taskPending=true;}
''' + 'function manualNextKey(event){' + handler + '''
function key(changes={}){return {code:'KeyN',repeat:false,target:{closest:()=>editing},preventDefault(){},...changes};}
manualNextKey(key());assert.equal(calls,1);
manualNextKey(key());assert.equal(calls,1);taskPending=false;
for(const change of [{repeat:true},{ctrlKey:true},{code:'Space'}])manualNextKey(key(change));
editing=true;manualNextKey(key());editing=false;
dialog=true;manualNextKey(key());dialog=false;
agentState.mode='auto';manualNextKey(key());agentState.mode='manual';
disabled=true;manualNextKey(key());disabled=false;
for(const state of ['READY','FAULT','SWITCHING']){taskState.execution_state=state;manualNextKey(key());}
taskState.execution_state='RUNNING';taskState.requested_stage='harvest_1';manualNextKey(key());
assert.equal(calls,1);
taskState.active_stage='harvest_1';manualNextKey(key());assert.equal(calls,2);
'''
    result = subprocess.run([node, '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, json.dumps(dict(stdout=result.stdout, stderr=result.stderr))
