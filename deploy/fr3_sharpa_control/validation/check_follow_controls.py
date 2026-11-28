"""Run real browser handlers against delayed fake RPCs; no browser/hardware needed."""
from pathlib import Path
import subprocess
from deploy.fr3_sharpa_control.ui import HTML


def main():
    script=HTML.split('let daggerState=null,daggerPending=false;',1)[1].split('let homePedalDown=',1)[0]
    harness=r'''
const assert=require('node:assert/strict');
let daggerState=null,daggerPending=false,executionActive=true;
let taskState=null,backend=null,calls=[],followReply=null,holdReply=null,statuses=[];
let editing=false,dialog=false,refreshes=0,safetyReject=false;
const nodes=new Map();const $=id=>{if(!nodes.has(id))nodes.set(id,{disabled:false,textContent:'',value:''});return nodes.get(id)};
const document={querySelector:()=>dialog,addEventListener(){}};const window={addEventListener(){}};
const defer=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject}};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const waitFor=async condition=>{for(let i=0;i<100&&!condition();i++)await new Promise(r=>setTimeout(r,5));assert.ok(condition(),'condition did not complete')};
async function refresh(){refreshes++;return new Promise(()=>{})} // Heavy refresh never finishes.
async function taskCommand(){};async function stop(){}
async function api(path,args){
 if(path==='dagger-status'){if(statuses.length)backend=statuses.shift();return {execution_running:executionActive,dagger:{...backend}}}
 assert.equal(path,'dagger-command');calls.push({...args});
 if(args.command==='follow'&&safetyReject){safetyReject=false;backend={...backend,mode:'HOLDING'};throw Error('Measured hold changed; wait for stability again')}
 const result={...backend,communication_error:undefined,sequence:backend.sequence+1,
  generation:backend.generation+(args.command==='hold'?1:0),mode:args.command==='hold'?'HOLDING':'HUMAN_LEFT'};
 const delay=args.command==='follow'?followReply:holdReply;
 if(delay){if(args.command==='follow')followReply=null;else holdReply=null;await delay.promise}
 backend=result;return {...result};
}
function reset(mode,extra={}){
 assert.equal(followBusy,false);
 backend={run_id:'test-run',generation:5,sequence:10,mode,input_ready:true,...extra};
 daggerState=null;daggerPending=false;executionActive=true;calls=[];statuses=[];followReply=holdReply=null;
 followPedalsDown.clear();renderDagger({...backend});
}
function key(type='keydown',extra={}){return {code:'KeyA',type,repeat:false,target:{closest:()=>editing},preventDefault(){},...extra}}
function tap(){followPedalKey(key());followPedalKey(key('keyup'))}
'''
    scenarios=r'''
(async()=>{
 // Release + second press is not lost while follow acknowledgement is pending.
 reset('HOLD');const delayed=defer();followReply=delayed;tap();await tick();
 assert.deepEqual(calls.map(x=>x.command),['follow']);assert.equal(followBusy,true);
 assert.equal($('daggerFollow').disabled,false);tap();assert.equal(followIntent.want,false);
 delayed.resolve();await waitFor(()=>!followBusy);
 assert.deepEqual(calls.map(x=>x.command),['follow','hold']);assert.equal(daggerState.mode,'HOLDING');
 assert.ok(refreshes>0); // The uncompleted full refresh did not keep controls locked.

 // Mouse cancellation uses exactly the same latest intent as the pedal.
 reset('HOLD');const mouse=defer();followReply=mouse;toggleFollowing();await tick();
 toggleFollowing();mouse.resolve();await waitFor(()=>!followBusy);
 assert.deepEqual(calls.map(x=>x.command),['follow','hold']);

 // A quick hold->follow request waits for actual confirmed hold before granting authority.
 reset('HUMAN_LEFT');const hold=defer();holdReply=hold;tap();await tick();tap();
 statuses=[{...backend,mode:'HOLDING',generation:6,sequence:12},{...backend,mode:'HOLD',generation:6,sequence:13}];
 hold.resolve();await waitFor(()=>!followBusy);
 assert.deepEqual(calls.map(x=>x.command),['hold','follow']);assert.equal(calls[1].expected_generation,6);

 // Another press cancels the waiting request; no delayed following is granted.
 reset('HOLDING');tap();await tick();assert.equal(followIntent.want,true);tap();
 await waitFor(()=>!followBusy);assert.equal(calls.length,0);
 renderDagger({...backend,mode:'HOLD',sequence:30});await tick();assert.equal(calls.length,0);

 // Repeat events and held-key duplicate keydowns never toggle twice.
 reset('HOLD');const repeat=defer();followReply=repeat;
 followPedalKey(key());followPedalKey(key('keydown',{repeat:true}));followPedalKey(key());
 assert.equal(followIntent.want,true);followPedalKey(key('keyup'));tap();
 repeat.resolve();await waitFor(()=>!followBusy);assert.deepEqual(calls.map(x=>x.command),['follow','hold']);

 // A stale full-page reply cannot overwrite a newer accepted control snapshot.
 reset('HUMAN_LEFT');renderDagger({...backend,mode:'HOLD',sequence:9});assert.equal(daggerState.mode,'HUMAN_LEFT');
 renderDagger({...backend,mode:'HOLD',generation:4,sequence:100});assert.equal(daggerState.mode,'HUMAN_LEFT');

 // A transient polling error may not disable the stop-follow action.
 reset('HUMAN_LEFT',{communication_error:'temporary status timeout'});
 assert.equal($('daggerFollow').disabled,false);tap();await waitFor(()=>!followBusy);
 assert.deepEqual(calls.map(x=>x.command),['hold']);

 // Failed/ambiguous follow reply: preserve a queued cancellation using fresh status.
 reset('HOLD');const failed=defer();followReply=failed;tap();await tick();tap();
 backend={...backend,mode:'HUMAN_LEFT',sequence:11};failed.reject(Error('reply lost'));
 await waitFor(()=>!followBusy);assert.deepEqual(calls.map(x=>x.command),['follow','hold']);

 // An ended/replaced run never inherits a pending follow/hold action.
 reset('HOLD');const replaced=defer();followReply=replaced;tap();await tick();tap();
 renderDagger({...backend,run_id:'new-run',mode:'HOLD',sequence:1});replaced.resolve();
 await waitFor(()=>!followBusy);assert.deepEqual(calls.map(x=>x.command),['follow']);
 assert.equal(daggerState.run_id,'new-run');

 // Waiting expires without ever sending follow, and does not restart later.
 reset('HOLDING');const realNow=Date.now;let now=realNow();Date.now=()=>now;
 tap();await tick();now+=7000;await waitFor(()=>!followBusy);Date.now=realNow;
 assert.equal(calls.length,0);assert.match($('daggerError').textContent,/保持确认超时/);

 // An explicit measured-hold rejection waits for safe confirmation, not a new pedal tap.
 reset('HOLD');safetyReject=true;
 statuses=[{...backend,mode:'HOLDING',sequence:11},{...backend,mode:'HOLD',sequence:12}];
 tap();await waitFor(()=>!followBusy);
 assert.deepEqual(calls.map(x=>x.command),['follow','follow']);assert.equal(daggerState.mode,'HUMAN_LEFT');

 // Inputs, reserved C, modifiers and a dialog never grant following.
 reset('HOLD');editing=true;tap();editing=false;dialog=true;tap();dialog=false;
 followPedalKey(key('keydown',{code:'KeyC'}));followPedalKey(key('keyup',{code:'KeyC'}));
 followPedalKey(key('keydown',{ctrlKey:true}));assert.equal(calls.length,0);
 console.log('12 real-handler regression scenarios passed; no hardware started');
})().catch(error=>{console.error(error);process.exitCode=1});
'''
    subprocess.run(['node','-'],input=harness+script+scenarios,text=True,check=True)


if __name__=='__main__':main()
