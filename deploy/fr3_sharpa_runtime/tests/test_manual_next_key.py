"""Stage selection remains available alongside automatic Agent control."""
import json
import shutil
import subprocess

import pytest

from deploy.fr3_sharpa_control.ui import HTML


def test_removed_controls_and_next_stage_shortcut():
    for name in ("agentMode", "taskResume", "taskRecover", "taskNext", "taskComplete"):
        assert f'id="{name}"' not in HTML
    assert "manualNextKey" not in HTML
    for name in ("taskStart", "taskPause", "taskStages"):
        assert f'id="{name}"' in HTML


def test_manual_selection_preempts_auto_and_queues_latest_click():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required")
    script = HTML.split("<script>", 1)[1].split("(async()=>{let d=await api('info')", 1)[0]
    harness = """
const assert=require('node:assert/strict');
const nodes=new Map(),calls=[];
function element(){
 return {textContent:'',value:'',checked:false,disabled:false,children:[],
  addEventListener(){},replaceChildren(){this.children=[]},append(c){this.children.push(c)},
  setAttribute(k,v){this[k]=v}};
}
const document={addEventListener(){},createElement:element,
 getElementById(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)}};
const window={addEventListener(){}};
let releaseStage=null,delayNext=false;
async function fetch(path,request){
 const payload=JSON.parse(request.body);
 if(path==='/api/preferences')return {ok:true,json:async()=>payload};
 calls.push([path,payload]);
 if(delayNext){delayNext=false;await new Promise(resolve=>releaseStage=resolve)}
 return {ok:true,json:async()=>({...taskState,requested_stage:payload.target,
  stage_version:taskState.stage_version+1,execution_state:'SWITCH_REQUESTED'})};
}
"""
    scenario = """
(async()=>{
 $('task_mode').value='tomato-in-hand119';$('agentProvider').value='tairan';
 agentState={status:'ready',mode:'auto',provider:'tairan'};
 taskModeChanged();
 $('taskStages').children[2].onclick();
 assert.equal(config().initial_stage,'adjust_in_hand');

 executionActive=true;
 const running={run_id:'one',stage_version:0,task_profile:'tomato-in-hand119',
 execution_state:'RUNNING',active_stage:'harvest_1',requested_stage:'harvest_1'};
 renderTask(running);
 assert.equal($('taskStages').children.every(b=>!b.disabled),true);
 delayNext=true;
 const first=selectLiveStage('return_stem');
 await Promise.resolve();
 selectLiveStage('adjust_in_hand');
 selectLiveStage('pickup');
 releaseStage();await first;
 assert.deepEqual(calls.map(c=>c[1].target),['return_stem','pickup']);
 assert.equal(calls[1][1].expected_version,1);
 assert.equal(calls[1][1].expected_run_id,'one');

 renderTask({...running,stage_version:2,execution_state:'PAUSED',safe_to_resume:true});
 assert.equal($('taskStart').disabled,false);
 assert.equal($('taskStart').textContent,'继续当前阶段');
 renderTask({...running,stage_version:3,execution_state:'PAUSED',safe_to_resume:false});
 assert.equal($('taskStart').disabled,true);

 renderTask({...running,stage_version:4,dagger_suspended:true});
 assert.equal($('taskStages').children.every(b=>!b.disabled),true);
 renderTask({...running,stage_version:5,execution_state:'FAULT'});
 $('taskStages').children[4].onclick();
 assert.equal(config().initial_stage,'return_stem');
 assert.equal($('taskStart').disabled,true);
 executionActive=false;
 $('task_mode').value='none';taskModeChanged();
 assert.equal($('taskPanel').hidden,true);
 assert.equal($('taskLayout').hidden,false);
})().catch(e=>{console.error(e);process.exitCode=1});
"""
    result = subprocess.run([node, "-"], input=harness + script + scenario, text=True, capture_output=True)
    assert result.returncode == 0, json.dumps(dict(stdout=result.stdout, stderr=result.stderr))
