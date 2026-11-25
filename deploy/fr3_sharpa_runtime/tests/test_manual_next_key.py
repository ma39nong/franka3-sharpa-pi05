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
    assert "disabled" not in options[1]
    assert "initial_stage" not in parser.selects
    assert [option["value"] for option in parser.selects["agentProvider"][1]] == ["mengxiang", "tairan"]



def test_prestart_stage_selection_and_deferred_agent_mode():
    from deploy.fr3_sharpa_control.ui import HTML

    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for UI state transitions")
    script = HTML.split("<script>", 1)[1].split("(async()=>{let d=await api('info')", 1)[0]
    harness = """
const assert=require('node:assert/strict');
const nodes=new Map(), calls=[];
function element(){
 return {
  textContent:'',value:'',checked:false,disabled:false,children:[],
  addEventListener(){},
  replaceChildren(){this.children=[]},
  append(child){this.children.push(child)},
  setAttribute(name,value){this[name]=value},
 };
}
const document={
 addEventListener(){},
 createElement:element,
 getElementById(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)},
};
const window={addEventListener(){}};
async function fetch(path, request){
 const payload=JSON.parse(request.body);
 if(path==='/api/preferences')return {ok:true,json:async()=>payload};
 calls.push([path,payload]);
 return {ok:true,json:async()=>({...agentState,mode:payload.mode})};
}
"""
    scenario = """
(async()=>{
 $('task_mode').value='tomato-in-hand119';
 $('agentProvider').value='tairan';
 agentState={mode:'manual',provider:'tairan',status:'waiting'};
 taskModeChanged();
 const stages=$('taskStages');
 assert.equal(stages.children.length,5);
 assert.equal(stages.children[0]['aria-pressed'],'true');
 assert.equal($('taskStart').disabled,true);

 approved='previous preview';
 $('recovery_confirmed').checked=true;
 stages.children[1].onclick();
 assert.equal(config().initial_stage,'harvest_1');
 assert.equal(stages.children[1]['aria-pressed'],'true');
 assert.equal($('recovery_confirmed').checked,false);
 assert.equal(approved,null);
 assert.deepEqual(calls,[]);

 $('agentMode').value='auto';
 await agentModeChanged();
 assert.equal($('agentMode').value,'auto');
 assert.equal($('agentMode').disabled,false);
 assert.deepEqual(calls,[]);

 agentState.status='ready';
 const ready={run_id:'new',stage_version:0,task_profile:'tomato-in-hand119',
              execution_state:'READY',requested_stage:'harvest_1'};
 renderTask(ready);
 await applyPendingAgentMode();
 assert.equal($('taskStart').disabled,true);
 assert.deepEqual(calls,[]);

 executionActive=true;
 renderTask({...ready,communication_error:'waiting'});
 await applyPendingAgentMode();
 assert.equal($('taskStart').disabled,true);
 assert.deepEqual(calls,[]);

 renderTask(ready);
 assert.equal($('taskStart').disabled,false);
 assert.equal(stages.children[1].disabled,false);
 await applyPendingAgentMode();
 await applyPendingAgentMode();
 assert.equal(agentState.mode,'auto');
 assert.deepEqual(calls,[['/api/agent-mode',{mode:'auto'}]]);

 // A backend fallback after activation must stay manual.
 renderAgent({...agentState,mode:'manual'});
 await applyPendingAgentMode();
 assert.equal($('agentMode').value,'manual');
 assert.equal(calls.length,1);

 // Manual control allows every stage while the Agent remains ready.
 renderTask({...ready,execution_state:'RUNNING',active_stage:'harvest_1'});
 assert.equal(agentState.status,'ready');
 assert.equal(stages.children.every(b=>!b.disabled),true);
 const originalTaskCommand=taskCommand,selected=[];
 taskCommand=(op,target)=>selected.push([op,target]);
 stages.children[4].onclick();
 stages.children[0].onclick();
 assert.deepEqual(selected,[['stage','return_stem'],['stage','pickup']]);
 taskCommand=originalTaskCommand;
 renderTask({...ready,execution_state:'RUNNING',dagger_suspended:true});
 assert.equal(stages.children.every(b=>b.disabled),true);

 executionActive=false;
 renderTask({...ready,execution_state:'FAULT'});
 assert.equal(stages.children[1]['aria-pressed'],'true');
 assert.equal($('taskStart').disabled,true);
 stages.children[0].onclick();
 assert.equal(config().initial_stage,'pickup');

 $('agentMode').value='auto';
 await agentModeChanged();
 $('agentMode').value='manual';
 await agentModeChanged();
 executionActive=true;
 renderTask(ready);
 await applyPendingAgentMode();
 assert.equal(calls.length,1);

 // A uniform-prompt policy has neither stage controls nor an Agent dependency.
 executionActive=false;agentState.mode='manual';
 $('task_mode').value='none';$('image_input').value='three';
 $('prompt').value='Harvest the whole truss into the basket';
 taskModeChanged();
 assert.equal($('unifiedPromptFields').hidden,false);
 assert.equal($('stageRecovery').hidden,true);
 assert.equal($('taskPanel').hidden,true);
 assert.equal($('taskLayout').hidden,false);
 assert.equal($('cropFields').hidden,true);
 assert.equal(config().prompt,'Harvest the whole truss into the basket');
 assert.equal(config().initial_stage,'pickup_1');
 assert.equal(calls.length,1);
 $('task_mode').value='tomato-in-hand119';$('image_input').value='four';
 taskModeChanged();
 assert.equal($('taskPanel').hidden,false);
 assert.equal($('unifiedPromptFields').hidden,true);
 assert.equal($('cropFields').hidden,false);
 assert.equal($('taskStages').children.length,5);
 assert.equal(calls.length,1);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run([node, "-"], input=harness + script + scenario,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
