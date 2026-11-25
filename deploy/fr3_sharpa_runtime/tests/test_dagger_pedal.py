"""Execute the real browser pedal handler without connecting robot hardware."""
import shutil
import subprocess

import pytest

from deploy.fr3_sharpa_control.ui import HTML


def test_pedal_toggles_confirmed_left_follow_and_hold_without_retrigger():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser pedal handler execution')
    handler = HTML.split('let daggerPedalDown=false;', 1)[1].split(
        "document.addEventListener('keydown',daggerPedalKey);", 1)[0]
    render = HTML.split('function renderDagger(d){', 1)[1].split(
        'async function daggerCommand(command){', 1)[0]
    script = """
const assert=require('node:assert/strict');
let daggerPedalDown=false,daggerState=null,daggerPending=false,executionActive=true;
let editing=false,dialog=false,prevented=0;
const nodes=new Map(),calls=[];
const $=id=>{if(!nodes.has(id))nodes.set(id,{});return nodes.get(id)};
const document={querySelector:()=>dialog};
function daggerCommand(command){calls.push(command);daggerPending=true;renderDagger(daggerState)}
""" + 'function renderDagger(d){' + render + handler + """
function key(changes={}){return {code:'KeyA',type:'keydown',repeat:false,
 target:{closest:()=>editing},preventDefault(){prevented++},...changes}}
function release(){daggerPedalKey(key({type:'keyup'}))}
function tap(changes={}){daggerPedalKey(key(changes));release()}
function state(mode,extra={}){daggerPending=false;renderDagger({mode,input_ready:true,...extra})}

// A never takes authority or resumes the model by itself.
for(const mode of ['IDLE','STARTING','POLICY','HOLDING','RESUMING','STOPPED']){
 state(mode);tap();
}
assert.deepEqual(calls,[]);
state('HOLD');daggerPedalKey(key());
assert.deepEqual(calls,['follow']);
// Repeat, duplicate keydown, and pending requests cannot toggle back.
daggerPedalKey(key({repeat:true}));daggerPedalKey(key());
state('HUMAN_LEFT');daggerPedalKey(key());
assert.deepEqual(calls,['follow']);release();
daggerPedalKey(key());assert.deepEqual(calls,['follow','hold']);release();
tap();assert.equal(calls.length,2); // request still pending

state('HOLD',{input_ready:false});tap();
state('HOLD',{communication_error:'offline'});tap();
state('HOLD');executionActive=false;tap();executionActive=true;
for(const modifier of ['ctrlKey','altKey','metaKey','shiftKey','isComposing'])tap({[modifier]:true});
editing=true;tap();editing=false;dialog=true;tap();dialog=false;
tap({target:{isContentEditable:true}});
tap({code:'KeyC'});tap({code:'KeyB'});tap({code:'KeyL'});tap({code:'Space'});
assert.equal(calls.length,2);
// Lost input must still permit hold from active human following.
state('HUMAN_LEFT',{input_ready:false});tap();
assert.deepEqual(calls,['follow','hold','hold']);
assert.equal(prevented,3);
state('HOLD');tap();assert.equal(calls.at(-1),'follow');
"""
    result = subprocess.run([node, '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
