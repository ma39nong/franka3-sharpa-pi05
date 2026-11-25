"""Run the actual rating UI against recorded lifecycle responses."""
import shutil
import subprocess

import pytest

from deploy.fr3_sharpa_control.ui import HTML


def test_dialog_reviews_failed_runs_and_notes_stay_bound_to_latest_episode():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required")
    script = HTML.split("let qualityRows=", 1)[1].split("let agentPresets=", 1)[0]
    harness = """
const assert=require('node:assert/strict');
const nodes=new Map();
function $(id){
 if(!nodes.has(id))nodes.set(id,{value:'',textContent:'',disabled:false,open:false,
 classList:{toggle(){}},addEventListener(){},showModal(){this.open=true},close(){this.open=false}});
 return nodes.get(id);
}
let executionActive=true,rows=[],requests=[];
async function api(route,data){
 if(route==='recordings')return structuredClone(rows);
 requests.push([route,data]);
 rows=rows.map(r=>r.run_id===data.run_id?{...r,grade:data.grade||r.grade,note:data.note,
 revision:r.revision+1,history:[{grade:data.grade||r.grade}]}:r);
}
$('output').value='/logs';
""" + "let qualityRows=" + script
    scenario = """
(async()=>{
 const failed={relative:'2026-11-25/19.00',run_id:'one',archive:{episode:'episode1'},
 can_rate:true,review_ready:true,has_recording:true,grade:'C',note:'auto',revision:1,
 history:[{automatic:true}],warnings:[],recording_status:'COMPLETE',
 execution:{outcome:'FAILED',reason:'input failed',ended_unix_ns:10}};
 rows=[failed];await refreshRecordings(true);
 assert.equal($('qualityDialog').open,false); // Cleanup/execution still active.
 executionActive=false;await refreshRecordings(true);
 assert.equal($('qualityDialog').open,true);
 assert.match($('qualityResult').textContent,/建议选择 C/);
 assert.equal($('qualityEpisode').textContent,'episode1');
 $('qualityNote').value='operator comment';
 await refreshRecordings(true);
 assert.equal($('qualityNote').value,'operator comment'); // Polling preserves draft.
 const newer={...failed,run_id:'two',relative:'2026-11-25/19.01',archive:{episode:'episode2'},
 grade:null,note:'',revision:0,history:[],execution:{outcome:'FINISHED',ended_unix_ns:20}};
 rows.push(newer);await refreshRecordings(true);
 await rateRecording('B');
 assert.equal(requests[0][1].run_id,'one'); // Open dialog keeps its original identity.
 assert.equal($('qualityDialog').open,true); // Next unreviewed episode follows.
 assert.equal($('qualityEpisode').textContent,'episode2');
 assert.match($('qualityResult').textContent,/正常结束/);
 await rateRecording('A');
 assert.equal($('qualityDialog').open,false);
 $('qualityNote').value='latest episode note';
 await submitQualityNote();
 assert.equal(requests.at(-1)[0],'recording-note');
 assert.equal(requests.at(-1)[1].run_id,'two');
 assert.equal(selectedRecording().note,'latest episode note');
 await refreshRecordings(true);
 assert.equal($('qualityDialog').open,false);
})().catch(e=>{console.error(e);process.exitCode=1});
"""
    result = subprocess.run([node, "-"], input=harness + scenario, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
