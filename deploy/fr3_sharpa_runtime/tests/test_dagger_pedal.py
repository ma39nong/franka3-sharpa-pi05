"""Exercise real pedal handlers and measured-hold recovery without hardware."""
import shutil
import subprocess

import pytest

from deploy.fr3_sharpa_control.ui import HTML


def test_space_takeover_and_resume_waits_for_hold_and_l_toggles_stage_stop():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser pedal handler execution')
    script = HTML.split('let daggerState=null,daggerPending=false;', 1)[1].split(
        'let homePedalDown=', 1)[0]
    harness = """
const assert=require('node:assert/strict');
let daggerState=null,daggerPending=false,executionActive=true;
let editing=false,dialog=false,taskState={run_id:'task-one',execution_state:'READY'};
const nodes=new Map(),calls=[];
const $=id=>{
 if(!nodes.has(id))nodes.set(id,{disabled:false,value:'',textContent:''});
 return nodes.get(id);
};
const document={querySelector:()=>dialog,addEventListener(){}};
const window={addEventListener(){}};
let statuses=[],stageAccepted=true;
async function api(route,values){
 if(route==='status')return statuses.shift();
 calls.push(values.command);
 return {...daggerState,mode:values.command==='hold'?'HOLDING':
  values.command==='resume'?'RESUMING':'HOLDING',generation:daggerState.generation+1};
}
async function refresh(){}
async function taskCommand(command){calls.push(command);return stageAccepted}
async function stop(){calls.push('stop')}
function state(mode,extra={}){
 daggerPending=false;
 renderDagger({run_id:'run-one',generation:1,mode,input_ready:true,...extra});
}
function key(code,extra={}){
 return {code,type:'keydown',repeat:false,target:{closest:()=>editing},
 preventDefault(){this.prevented=true},...extra};
}
"""
    scenario = """
(async()=>{
 state('HUMAN_LEFT');
 statuses=[{execution_running:true,dagger:{...daggerState,mode:'HOLDING'}},
  {execution_running:true,dagger:{...daggerState,mode:'HOLD',generation:2}}];
 await daggerCommand('resume');
 assert.deepEqual(calls,['hold','resume']);
 assert.equal(daggerState.mode,'RESUMING');
 assert.equal(daggerPending,false);

 calls.length=0;state('HOLD');
 await daggerCommand('resume');
 assert.deepEqual(calls,['resume']);

 calls.length=0;state('HUMAN_LEFT');
 statuses=[{execution_running:false,dagger:{...daggerState,mode:'STOPPED'}}];
 await daggerCommand('resume');
 assert.deepEqual(calls,['hold']); // Ended run never receives a resume request.
 assert.match($('daggerError').textContent,/恢复已取消/);

 // Stub dispatch only after exercising the real hold/confirmation sequence.
 daggerCommand=command=>{calls.push(command);daggerPending=true};
 calls.length=0;state('POLICY');
 const press=key('Space');daggerPedalKey(press);
 assert.equal(press.prevented,true);
 assert.deepEqual(calls,['takeover']);
 daggerPedalKey(key('Space',{repeat:true}));
 daggerPedalKey(key('Space'));
 assert.equal(calls.length,1);
 daggerPedalKey(key('Space',{type:'keyup'}));
 for(const mode of ['HOLD','HOLDING','HUMAN_LEFT']){
  state(mode);daggerPedalKey(key('Space'));daggerPedalKey(key('Space',{type:'keyup'}));
  assert.equal(calls.at(-1),'resume');
 }
 const before=calls.length;
 for(const mode of ['IDLE','STARTING','STOPPED']){
  state(mode);daggerPedalKey(key('Space'));daggerPedalKey(key('Space',{type:'keyup'}));
 }
 state('POLICY',{communication_error:'offline'});
 daggerPedalKey(key('Space'));daggerPedalKey(key('Space',{type:'keyup'}));
 assert.equal(calls.length,before);
 state('POLICY');
 for(const modifier of ['ctrlKey','altKey','metaKey','shiftKey','isComposing']){
  daggerPedalKey(key('Space',{[modifier]:true}));
 }
 editing=true;daggerPedalKey(key('Space'));editing=false;
 dialog=true;daggerPedalKey(key('Space'));dialog=false;
 daggerPedalKey(key('KeyA'));
 assert.equal(calls.length,before);

 // A follows only after confirmed takeover, then holds even if input is lost.
 calls.length=0;state('HOLD');
 followPedalKey(key('KeyA'));
 assert.deepEqual(calls,['follow']);
 followPedalKey(key('KeyA',{repeat:true}));
 followPedalKey(key('KeyA'));
 assert.equal(calls.length,1);
 followPedalKey(key('KeyA',{type:'keyup'}));
 state('HUMAN_LEFT',{input_ready:false});
 followPedalKey(key('KeyA'));
 assert.deepEqual(calls,['follow','hold']);
 followPedalKey(key('KeyA',{type:'keyup'}));
 state('HOLD',{input_ready:false});
 followPedalKey(key('KeyA'));followPedalKey(key('KeyA',{type:'keyup'}));
 for(const mode of ['POLICY','IDLE','HOLDING','RESUMING','STOPPED']){
  state(mode);followPedalKey(key('KeyA'));followPedalKey(key('KeyA',{type:'keyup'}));
 }
 state('HOLD');
 for(const modifier of ['ctrlKey','altKey','metaKey','shiftKey','isComposing']){
  followPedalKey(key('KeyA',{[modifier]:true}));
 }
 editing=true;followPedalKey(key('KeyA'));editing=false;
 dialog=true;followPedalKey(key('KeyA'));dialog=false;
 followPedalKey(key('KeyA',{target:{isContentEditable:true}}));
 followPedalKey(key('KeyC'));followPedalKey(key('KeyC',{type:'keyup'}));
 assert.deepEqual(calls,['follow','hold']);
 state('HOLD',{communication_error:'offline'});
 followPedalKey(key('KeyA'));followPedalKey(key('KeyA',{type:'keyup'}));
 state('HOLD');executionActive=false;followPedalKey(key('KeyA'));executionActive=true;
 assert.deepEqual(calls,['follow','hold']);

 calls.length=0;$('task_mode').value='tomato-in-hand119';
 await taskPedalKey(key('KeyL'));
 assert.deepEqual(calls,['start']);
 await taskPedalKey(key('KeyL',{repeat:true}));
 await taskPedalKey(key('KeyL'));
 assert.equal(calls.length,1);
 await taskPedalKey(key('KeyL',{type:'keyup'}));
 await taskPedalKey(key('KeyL'));
 assert.deepEqual(calls,['start','stop']); // Even while READY feedback lags.
 await taskPedalKey(key('KeyL',{type:'keyup'}));
 for(const modifier of ['ctrlKey','altKey','metaKey','shiftKey','isComposing']){
  await taskPedalKey(key('KeyL',{[modifier]:true}));
  await taskPedalKey(key('KeyL',{type:'keyup'}));
 }
 editing=true;await taskPedalKey(key('KeyL'));editing=false;
 await taskPedalKey(key('KeyL',{type:'keyup'}));
 dialog=true;await taskPedalKey(key('KeyL'));dialog=false;
 await taskPedalKey(key('KeyL',{type:'keyup'}));
 executionActive=false;await taskPedalKey(key('KeyL'));executionActive=true;
 await taskPedalKey(key('KeyL',{type:'keyup'}));
 $('task_mode').value='none';await taskPedalKey(key('KeyL'));
 assert.deepEqual(calls,['start','stop']);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run([node, '-'], input=harness + script + scenario,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_both_end_buttons_use_the_same_stop_action():
    from html.parser import HTMLParser

    class Buttons(HTMLParser):
        def __init__(self):
            super().__init__()
            self.buttons = {}
        def handle_starttag(self, tag, attributes):
            values = dict(attributes)
            if tag == 'button' and values.get('id'):
                self.buttons[values['id']] = values

    parser = Buttons()
    parser.feed(HTML)
    assert parser.buttons['daggerEnd']['onclick'] == 'stop()'
    assert parser.buttons['daggerTakeoverRight']['disabled'] is None
    assert parser.buttons['daggerFollowRight']['disabled'] is None
    assert 'id="daggerHold"' not in HTML
    assert '<button class="stop" onclick="stop()">停止执行</button>' in HTML
