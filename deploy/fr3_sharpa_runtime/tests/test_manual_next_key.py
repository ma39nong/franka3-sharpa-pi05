import json
from pathlib import Path
import shutil
import subprocess
from html.parser import HTMLParser

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


def test_agent_auto_option_stays_in_stage_mode_dropdown():
    from deploy.fr3_sharpa_control.ui import HTML

    class SelectOptions(HTMLParser):
        def __init__(self):
            super().__init__()
            self.selects = {}
            self.active = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "select":
                self.active = attrs.get("id")
                self.selects[self.active] = (attrs, [])
            if tag == "option" and self.active:
                self.selects[self.active][1].append(attrs)

        def handle_endtag(self, tag):
            if tag == "select":
                self.active = None

    parser = SelectOptions()
    parser.feed(HTML)
    mode, options = parser.selects["agentMode"]
    assert "disabled" not in mode
    assert [option["value"] for option in options] == ["manual", "auto"]
    assert "disabled" in options[1]
    assert [option["value"] for option in parser.selects["agentProvider"][1]] == ["mengxiang", "tairan"]

    node = shutil.which("node")
    if not node:
        return
    start = HTML.index("function updateAgentModeControl()")
    end = HTML.index("function renderAgent(a)", start)
    script = """
const assert=require('node:assert/strict');
const nodes={agentMode:{disabled:false},agentAutoMode:{},agentProvider:{value:'tairan'},
             agentModeHint:{},task_mode:{value:'tomato-in-hand119'}};
const $=id=>nodes[id];
let agentState={mode:'manual',provider:'tairan',status:'waiting'},taskState=null;
""" + HTML[start:end] + """
updateAgentModeControl();
assert.equal(nodes.agentMode.disabled,false);
assert.equal(nodes.agentAutoMode.disabled,true);
agentState.status='ready';
taskState={task_profile:'tomato-in-hand119',execution_state:'READY'};
updateAgentModeControl();
assert.equal(nodes.agentAutoMode.disabled,false);
agentState.provider='mengxiang';
updateAgentModeControl();
assert.equal(nodes.agentAutoMode.disabled,true);
assert.equal(nodes.agentMode.disabled,false);
"""
    subprocess.run([node, "-"], input=script, text=True, check=True)
