"""Loopback-only operator UI. No devices are started by opening the page."""
import argparse
from contextlib import suppress
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import signal
import threading
from urllib.parse import parse_qs, urlparse

from .controller import DeploymentController
from .plan import desktop, inspect_checkpoint, repository_info

HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sharpa 部署控制台</title><style>
*{box-sizing:border-box}body{margin:0;background:#eff3f6;color:#15293e;font:14px/1.55 system-ui,sans-serif}
header{background:white;color:#123a56;border-bottom:1px solid #dae3e9;padding:6px 24px;display:flex;justify-content:space-between;align-items:center;gap:16px}
.brand{flex:1;min-width:0}.brand-logo{display:block;width:260px;max-width:100%;height:72px;background:#123a56;-webkit-mask:url('/assets/liqing-logo.png') center/contain no-repeat;mask:url('/assets/liqing-logo.png') center/contain no-repeat}

main{width:100%;padding:18px 20px 14px}.dashboard-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-bottom:16px;align-items:stretch}
.card{min-width:0;background:white;border:1px solid #dae3e9;border-radius:10px;padding:18px}h2{font-size:17px;margin:0 0 14px}label{display:block;margin-top:11px;font-weight:600}input,select{width:100%;min-width:0;padding:9px;border:1px solid #b9c8d2;border-radius:6px;background:white;font:inherit;color:inherit}
input[type=checkbox]{width:auto}.row{display:flex;gap:10px;align-items:end}.row>*{flex:1;min-width:0}.path{display:flex;gap:6px}.path input{min-width:0}.path button{flex:0 0 auto}
button{border:0;border-radius:6px;padding:9px 12px;background:#dfe9ee;color:#17394a;font:600 13px system-ui;cursor:pointer;white-space:nowrap}button.primary{background:#087b75;color:white}button.stop{background:#b63f37;color:white}button:disabled{opacity:.45;cursor:not-allowed}
.muted{color:#62778a;font-size:12px}.notice{background:#e9f5f3;padding:10px;border-radius:6px;margin:11px 0;overflow-wrap:anywhere}.error{color:#b7352f;white-space:pre-wrap;overflow-wrap:anywhere}.badge{padding:5px 11px;background:#eef3f6;color:#123a56;border-radius:20px;font-size:12px;white-space:nowrap}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,monospace;margin:8px 0;max-height:250px;overflow:auto}.control-line{padding:9px 0;border-bottom:1px solid #e5edf1}.control-line>span{margin-left:10px}.control-line .actions{margin-top:9px}.actions{display:flex;gap:8px;flex-wrap:wrap}
.logs .card{padding:14px;display:flex;flex-direction:column}.logs h2{font-size:15px;margin-bottom:10px}.logs pre{height:clamp(180px,21vh,270px);max-height:none;background:#132b38;color:#d5e5ed;padding:12px;border-radius:6px;margin:0;overflow:auto}.logs .is-active{border-color:#7bb5b0;box-shadow:0 0 0 1px #7bb5b0}
.page-note{margin:0;color:#62778a;font-size:12px}#directory{overflow-wrap:anywhere}#recordStatus{margin-top:6px}
dialog{border:0;border-radius:10px;width:760px;max-width:95vw}#entries{height:350px;overflow:auto}#entries button{display:block;background:white;text-align:left;width:100%;border-bottom:1px solid #e5edf1}details{margin:11px 0}summary{cursor:pointer;font-weight:600}.preview{min-height:60px}
@media(max-width:1100px){.dashboard-row{grid-template-columns:1fr}.logs pre{height:220px}header{padding:6px 18px}main{padding:14px}}
@media(max-width:520px){.brand-logo{height:60px;width:220px}.card{padding:14px}.row{flex-wrap:wrap}.row>*{flex-basis:42%}}
.task-stages{display:flex;gap:8px;flex-wrap:wrap}.task-stages .active{background:#087b75;color:white}.task-stages .pending{outline:2px solid #d69e36}.task-stages .passed{border-bottom:3px solid #7bb5b0}.task-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:6px}#taskPanel{margin-bottom:16px}#taskError{color:#b7352f}</style><header><div class="brand"><span class="brand-logo" role="img" aria-label="厘清智能"></span></div><span id="phase" class="badge">空闲</span></header>
<main><section class="dashboard-row configuration" aria-label="部署配置与控制"><article class="card" id="weights-card"><h2>1 · 权重与代码</h2>
<label>部署仓库目录</label><div class="path"><input id="repository"><button onclick="browse('repository')">选择目录</button></div>
<div class="muted" id="repoInfo">Main 与 Soft 可各用一个部署目录；不会在运行中切换 Git 分支。</div>
<label>Checkpoint 步骤目录</label><div class="path"><input id="checkpoint" placeholder="直接包含 params/ 和 assets/ 的目录"><button onclick="browse('checkpoint')">选择目录</button></div>
<label>模型类型（手动选择）</label><select id="model_type"><option value="full">Full Absolute · 全量绝对位置</option><option value="full-delta">Full Delta · 双臂增量</option><option value="lora">LoRA Absolute</option><option value="paper-delta">Paper Delta</option><option value="train-rtc-absolute">Train RTC Absolute</option><option value="train-rtc-delta">Train RTC Delta</option><option value="train-paper-rtc">Train Paper RTC</option><option value="auto">兼容：按现有训练记录自动识别</option></select>
<div class="muted">直接选择checkpoint及其真实训练类型，无需生成、复制部署契约。仍会检查权重结构和配套统计；Paper使用原有算法资产。</div>
<details><summary>自动识别时的动作语义</summary><select id="semantics"><option value="auto">自动识别</option><option value="absolute">Absolute</option><option value="delta">Delta</option></select></details>
<details><summary>训练证据（可选，用于交叉核对）</summary><label>归档训练 JSON（可选）</label><div class="path"><input id="training_record"><button onclick="browse('training_record',true)">选择文件</button></div></details>
<label>训练 RTC 最大延迟（0 自动识别）</label><input id="train_rtc_max_delay" type="number" min="0" max="49" value="0"><div class="muted">手动选择Train RTC时必填实际训练上限（例如10）；普通RTC无需填写。训练JSON仅用于可选核对，不会生成契约副本。</div>
<label>Train RTC 初始延迟（0 使用训练上限，最多25）</label><input id="rtc_initial_delay_steps" type="number" min="0" max="25" value="0">
<div class="actions" style="margin-top:14px"><button onclick="inspect()">识别权重</button><button onclick="tomatoPreset()">使用本次 B 模型推荐配置</button></div><div id="identity" class="notice">覆盖：30Hz／58D 的全量 Absolute、普通 Delta、Paper Delta、Train RTC、Train Paper RTC，以及 Absolute LoRA。Paper按资产识别；LoRA Delta与其他结构尚不支持。实际结构在GPU加载时再次核对。</div>
</article><article class="card" id="execution-card"><h2>2 · 本次执行配置</h2>
<div class="row"><div><label>执行模式</label><select id="mode" onchange="modeChanged()"><option value="serial">普通串行</option><option value="rtc">RTC</option><option value="train-rtc">Train RTC · 训练前缀条件</option><option value="train-paper-rtc">Train Paper RTC · 训练前缀＋Paper</option><option value="paper-rtc">Paper RTC · 时间步归一化</option><option value="soft-serial">串行 Soft</option><option value="soft-async">异步 Soft · 50→30→26＋4→20</option></select></div><div><label>启动姿态</label><select id="initialization"><option value="current">Current · 从当前状态推理</option><option value="home">Home · 先回固定位置再推理</option></select></div></div>
<div class="row"><div><label>速度</label><select id="speed"><option value="slow">Slow · 0.1倍</option><option value="medium">Medium · 0.5倍</option><option value="fast">Fast · 模型原速</option></select></div><div><label id="stepsLabel">每轮执行的模型动作数</label><input id="steps" type="number" value="15" min="1" max="50"></div><div><label>轮数（0为持续）</label><input id="rounds" type="number" value="0" min="0"></div></div>
<div id="modeHint" class="notice"></div>
<label>启动前相机策略</label><select id="camera_policy"><option value="exclusive">部署独占接管：停止同组已知相机服务</option><option value="require_free">要求相机空闲：发现占用即报错</option></select>
<div class="notice">独占接管会停止同组 teleop-cameras／openpi-cameras，旧遥操预览或录制会中断。只在启动前操作；未知占用不杀进程，运行中不重启相机。支持并行启动的Home模式会在视频验证期间开始回位；两边均成功后才开始任务。</div><label><input id="record" type="checkbox"> 录制 ROSbag，并在结束后生成三路及三联视频</label>
<label>任务控制</label><select id="task_mode"><option value="none">普通部署 · 无阶段状态机</option><option value="tomato-b">番茄 B Coarse · 七阶段人工控制＋普通 RTC</option></select>
<label>任务初始阶段</label><select id="initial_stage"><option value="pickup_1">首次拿起</option><option value="harvest_1">首次采摘</option><option value="return_truss">放回整串</option><option value="adjust">调整</option><option value="pickup_2">再次拿起</option><option value="harvest_2">再次采摘</option><option value="return_stem">放回裸杆</option></select>
<label><input type="checkbox" id="recovery_confirmed"> 人工恢复：已停止旧执行、检查设备，明确授权从选定阶段重新开始</label>
<div class="muted">非首次阶段必须勾选并重新核对配置。恢复会获取新观测和首块，不复用旧动作。</div>
<label>归档父目录</label><div class="path"><input id="output"><button onclick="browse('output')">选择目录</button></div><div class="muted">自动创建 日期/小时.分钟（如15.43）；同一分钟重启自动加后缀，不覆盖旧记录。</div>
</article><article class="card" id="control-card"><h2>3 · 推理与执行</h2>
<div class="control-line"><strong>推理端</strong><span id="modelState" class="muted">未启动</span><div class="actions"><button id="startModel" class="primary" onclick="startModel()">启动推理</button><button id="closeModel" class="stop" onclick="closeModel()" disabled>停止推理</button></div></div>
<div class="control-line"><strong>执行端</strong><span id="executionState" class="muted">未运行</span><div class="actions"><button id="preview" onclick="preview()">核对执行配置</button><button id="start" class="primary" onclick="start()" disabled>启动执行</button><button id="stop" class="stop" onclick="stop()" disabled>停止执行</button></div></div>
<p class="muted">先启动推理，预热就绪后再核对并启动执行。番茄任务执行端就绪后，还需点击下方“开始当前阶段”。执行自动完成硬件 up；停止执行等效一次 Ctrl-C，等待失能确认及硬件清理，推理端保持运行。停止后可重新选择 Current/Home，再次执行自动 up。</p><div id="message" class="notice">尚未启动设备</div><div id="error" class="error"></div><details id="planDetails"><summary id="previewSummary">执行配置明细</summary><pre id="previewText" class="preview">请选择配置后点击“核对执行配置”。</pre></details><div id="directory" class="muted"></div><div id="recordStatus" class="muted"></div></article></section>
<section class="card" id="taskPanel" aria-label="番茄采摘任务状态机"><h2>番茄采摘任务状态机</h2>
<div id="taskStages" class="task-stages"></div><details><summary>训练四种英文指令（固定原文）</summary><pre>pickup: Right hand grasps and lifts the truss to the center
harvest: Right hand holds truss while left hand picks tomatoes into left basket
return: Right hand returns the held object to the right plate
adjust: Right fingers reposition the truss on the right plate</pre></details><p class="muted">阶段由人工推进。“已离开”仅表示人工切换，不代表模型验收成功。请求阶段和实际阶段可能不同。</p>
<div id="taskStatus" class="task-grid"><span>尚未启动番茄任务</span></div>
<details><summary>当前实际生效的英文 Prompt</summary><pre id="taskPrompt">尚未接入阶段动作</pre></details>
<div class="actions"><button id="taskStart" class="primary" onclick="taskCommand('start')" disabled>开始当前阶段</button><button id="taskPause" onclick="taskCommand('pause')" disabled>暂停</button><button id="taskResume" onclick="taskCommand('resume')" disabled>恢复（重新授权）</button><button id="taskNext" onclick="taskCommand('next')" disabled>切换到下一阶段</button><button id="taskRecover" onclick="recoverTask()" disabled>人工恢复／跳转</button><button id="taskComplete" onclick="taskCommand('complete')" disabled>完成任务</button><button class="stop" onclick="stop()">停止执行</button></div>
<p class="notice">跨阶段先停止并验证安全边界，再重新观测。当前硬件尚无双臂安全保持确认接口；切换／暂停将执行双手失能及完整清理，并阻止自动恢复。请人工检查后重新核对、启动。关闭本页面或任务通信失联超过5秒，也会停止任务。这些按钮属于软件控制，不是硬件急停。</p><div id="taskError" role="alert"></div></section>
<section class="dashboard-row logs" id="logs" aria-label="三个终端日志">
<article class="card" id="terminal-hardware"><h2>终端一 · 硬件日志</h2><pre id="log-hardware" role="log" aria-label="硬件日志">尚无硬件日志</pre></article>
<article class="card" id="terminal-inference"><h2>终端二 · 推理日志</h2><pre id="log-inference" role="log" aria-label="推理日志">尚无推理日志</pre></article>
<article class="card" id="terminal-execution"><h2>终端三 · 执行日志</h2><pre id="log-execution" role="log" aria-label="执行日志">尚无执行日志</pre></article>
</section>
<p class="page-note">打开页面不会连接设备；普通部署关闭页面不等于停止执行；番茄任务丢失页面心跳超过5秒会触发停止。请优先使用停止按钮。完整日志保存在归档目录。</p>
</main>
<dialog id="picker"><h2>选择工作机上的路径</h2><div class="path"><input id="browsePath"><button onclick="loadDirectory()">打开</button><button onclick="upDirectory()">上一级</button></div><div id="entries"></div><div class="actions"><button class="primary" id="chooseDirectory" onclick="choosePath(document.getElementById('browsePath').value)">选择此目录</button><button onclick="document.getElementById('picker').close()">取消</button></div></dialog>
<script>
let token='', approved=null, approvedPlan=null, approvedModelKey=null, busy=false, pickerTarget='', pickerFile=false, repoModes=['serial','rtc'], modelModes=null, previousMode='serial', repoTimer=null;
const $=id=>document.getElementById(id);
const fields=['repository','checkpoint','mode','initialization','speed','steps','rounds','record','output','training_record','semantics','camera_policy','train_rtc_max_delay','model_type','rtc_initial_delay_steps','task_mode','initial_stage','recovery_confirmed'];
function config(){let c={};for(let f of fields)c[f]=['record','recovery_confirmed'].includes(f)?$(f).checked:['steps','rounds','train_rtc_max_delay','rtc_initial_delay_steps'].includes(f)?Number($(f).value):$(f).value;return c}
async function api(path,data){const r=await fetch('/api/'+path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':token},body:JSON.stringify(data)});let d=await r.json();if(!r.ok)throw Error(d.error);return d}
function failure(e){$('error').textContent=e.message}
function invalidate(){approved=null;approvedPlan=null;approvedModelKey=null;$('previewSummary').textContent='执行配置明细';$('start').disabled=true;$('error').textContent=''}
function modeChanged(){let mode=$('mode').value,soft=mode.startsWith('soft-');$('steps').disabled=soft;$('steps').value=soft?26:15;$('steps').max=['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(mode)?24:50;$('stepsLabel').textContent=soft?'源执行点（固定26）':['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(mode)?'重规划触发源节点':'每轮执行的模型动作数';$('speed').disabled=['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(mode);if(['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(mode))$('speed').value='fast';else if(['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(previousMode))$('speed').value='slow';previousMode=mode;$('modeHint').textContent=mode==='soft-async'?'固定 50→30→26＋4→20；20是压缩后的目标数，不是重规划15点。':mode==='soft-serial'?'50→30→26＋4；26压成20，原串行压缩策略可能回退26。':['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(mode)?'原速后台重规划；此处节点数不是串行的固定执行前缀。':'每轮预测50，执行填写的前N个动作，然后重新观测推理。';invalidate()}
function modeNames(modes){return modes.map(m=>Array.from($('mode').options).find(o=>o.value===m).textContent).join('、')}
function updateModes(){let modes=repoModes.filter(m=>!modelModes||modelModes.includes(m));for(let o of $('mode').options)o.disabled=!modes.includes(o.value);if(!modes.includes($('mode').value)){$('mode').value=modes[0]||'serial';modeChanged()}if(modelModes)$('identity').textContent=$('identity').textContent.replace(/可用模式：.*/, '可用模式：'+modeNames(modes));return modes}
function modelTypeChanged(reset=false){let t=$('model_type').value,train=t.startsWith('train-');modelModes=t==='auto'?null:t==='lora'?['serial']:train?['serial',t==='train-paper-rtc'?'train-paper-rtc':'train-rtc']:['serial',t==='paper-delta'?'paper-rtc':'rtc','soft-serial','soft-async'];if(t!=='auto')$('semantics').value=t.includes('delta')||t==='train-paper-rtc'?'delta':'absolute';if(reset&&!train&&t!=='auto'){$('train_rtc_max_delay').value=0;$('rtc_initial_delay_steps').value=0}updateModes();invalidate()}
async function inspect(){try{const c=config();const d=await api('inspect',c);modelModes=d.supported_modes;const modes=updateModes();$('identity').textContent=`${d.tomato_model?d.tomato_model.name+' · ':''}${d.finetune_mode} · ${d.semantics||'语义未知'} · 30 Hz / 58D / horizon 50\n${d.evidence}${d.train_rtc?" · Train RTC 最大延迟="+d.train_rtc.max_delay+"步":""}\n归一化：${d.normalization}；噪声：${d.paper_algorithm?'相关噪声 β='+d.paper_algorithm.beta:'独立高斯'}\n可用模式：${modeNames(modes)}`;invalidate()}catch(e){failure(e)}}
async function repositoryChanged(){try{let d=await api('repository?path='+encodeURIComponent($('repository').value));repoModes=d.modes;$('repoInfo').textContent='当前分支：'+d.branch+'；仅可选当前代码支持的模式';updateModes();invalidate()}catch(e){failure(e)}}
async function preview(){try{const c=config();const p=await api('preview',c);$('previewText').textContent=JSON.stringify({分支:p.branch,提交:p.commit.slice(0,8),权重:p.model.checkpoint,模型:p.model.profile,证据:p.model.evidence,归一化:p.model.normalization,论文设置:p.model.paper_algorithm?{归一化:p.model.paper_algorithm.mode,相关噪声系数:p.model.paper_algorithm.beta}:null,方式:p.description,任务模式:c.task_mode,初始阶段:c.initial_stage,人工恢复确认:c.recovery_confirmed,姿态:c.initialization,速度:c.speed,轮数:c.rounds,录制:c.record,相机策略:c.camera_policy,归档:p.settings.output},null,2);approved=JSON.stringify(c);approvedPlan=p.approval;approvedModelKey=JSON.stringify([p.settings.repository,p.commit,p.model.identity,p.model.profile,p.port,p.settings.mode]);$('previewSummary').textContent='执行配置已核对 · 点击展开';$('error').textContent='';await refresh()}catch(e){invalidate();failure(e)}}
async function tomatoPreset(){if(busy)return;try{let info=await api('info');for(let [k,v]of Object.entries(info.tomato_preset)){if(fields.includes(k)){if(['record','recovery_confirmed'].includes(k))$(k).checked=v;else $(k).value=v}}modelModes=null;modelTypeChanged(true);modeChanged();invalidate();await repositoryChanged();await inspect()}catch(e){failure(e)}}
async function startModel(){try{const c=config();showLog('inference');await api('start-model',{settings:c});await refresh()}catch(e){failure(e)}}
function showLog(name){for(let key of ['hardware','inference','execution'])$('terminal-'+key).classList.toggle('is-active',key===name)}
async function start(){try{if(approved!==JSON.stringify(config()))throw Error('配置已改变，请重新核对');await api('start',{settings:config(),approval:approvedPlan});showLog('execution');busy=true;$('start').disabled=true;await refresh()}catch(e){failure(e)}}
async function stop(){try{await api('stop',{});await refresh()}catch(e){failure(e)}}
async function closeModel(){try{await api('close-model',{});await refresh()}catch(e){failure(e)}}
async function refresh(){try{let s=await api('status');busy=s.running;
renderTask(s.task);
$('phase').textContent=({idle:'空闲',validating:'配置检查',hardware_up:'启动硬件',model_loading:'推理预热中',model_stopping:'停止推理中',model_ready:'推理就绪',model_off:'推理已停止',recording_ready:'准备录制',running:'执行中',stopping:'停止执行中',failed:'失败',stopped:'执行已停止',finished:'执行已完成'})[s.phase]||s.phase;
$('message').textContent=s.message;$('modelState').textContent=s.model_loading?'正在加载／停止':s.model_ready?(s.model_owned?'已就绪':'已就绪（外部终端）'):s.model_running?'未就绪':'未启动';$('executionState').textContent=s.execution_running?'执行／清理中':'未运行';
$('stop').disabled=!s.execution_running||s.stop_requested;$('start').disabled=busy||!s.model_ready||JSON.stringify(s.model_key)!==approvedModelKey||approved!==JSON.stringify(config());$('startModel').disabled=busy||!$('checkpoint').value;$('closeModel').disabled=s.execution_running||(!s.model_owned&&!s.model_loading);$('preview').disabled=busy;
$('directory').textContent=s.directory?'本次日志目录：'+s.directory:'';$('recordStatus').textContent=[s['recording.json']?'录制：'+s['recording.json'].status:'',s['video_export.json']?'视频：'+s['video_export.json'].status:'',s.exports.length?'后台导出：'+s.exports.length+'项':''].filter(Boolean).join(' · ');
for(let key of ['hardware','inference','execution']){let pre=$('log-'+key),end=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-8;let text=s.logs[key]||({'hardware':'尚无硬件日志','inference':'尚无推理日志','execution':'尚无执行日志'})[key];if(pre.textContent!==text){pre.textContent=text;if(end)pre.scrollTop=pre.scrollHeight}}
for(let f of fields)$(f).disabled=busy;if(!busy){$('steps').disabled=$('mode').value.startsWith('soft-');$('speed').disabled=['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes($('mode').value)}}catch(e){failure(e)}}

let taskState=null, taskPending=false;
const stageList=[['pickup_1','首次拿起'],['harvest_1','首次采摘'],['return_truss','放回整串'],['adjust','调整'],['pickup_2','再次拿起'],['harvest_2','再次采摘'],['return_stem','放回裸杆']];
function renderTask(t){taskState=t;let state=t?.execution_state||'IDLE',available=!!t&&!t.communication_error,idx=stageList.findIndex(x=>x[0]===t?.requested_stage),canSwitch=available&&!taskPending&&['READY','RUNNING','SWITCH_REQUESTED','SWITCHING','RESUMING','PAUSED'].includes(state);let box=$('taskStages');box.replaceChildren();stageList.forEach(([id,name],i)=>{let b=document.createElement('button');b.textContent=(t?.active_stage===id?'● ':t?.requested_stage===id?'◷ ':'')+name+(t?.completed_stages?.includes(id)?' · 已离开':'');b.className=t?.active_stage===id?'active':t?.requested_stage===id?'pending':t?.completed_stages?.includes(id)?'passed':'';b.disabled=!canSwitch||i!==idx+1;b.onclick=()=>taskCommand('stage',id);box.append(b)});
let values={'请求阶段':t?.requested_stage||'—','实际阶段':t?.active_stage||'尚未接入','执行状态':t?.communication_error?'通信中断（上次 '+state+'）':state,'Prompt 类别':t?.stages?.find(x=>x.id===t.active_stage)?.action||'—','Stage Version':t?.stage_version??'—','Chunk ID':t?.active_chunk_id||'—','RTC 模式':t?.rtc_mode||'—','执行进度':t?.chunk_progress?Number(t.chunk_progress.phase).toFixed(1)+'/50':'—','在途推理':t?.inference_status||'—','最新请求':t?.request_status||'—','允许恢复':t?.safe_to_resume?'是，仍需明确授权':'否'};let status=$('taskStatus');status.replaceChildren();for(let [k,v]of Object.entries(values)){let span=document.createElement('span');span.textContent=k+'：'+v;status.append(span)}$('taskPrompt').textContent=t?.active_prompt||'尚未接入阶段动作';$('taskError').textContent=t?.communication_error||t?.last_error||'';
$('taskStart').disabled=!available||taskPending||state!=='READY';$('taskPause').disabled=!available||taskPending||!['RUNNING','RESUMING','SWITCHING','SWITCH_REQUESTED'].includes(state);$('taskResume').disabled=!available||taskPending||state!=='PAUSED'||!t?.safe_to_resume;$('taskNext').disabled=!canSwitch||idx>=6;$('taskRecover').disabled=!canSwitch;$('taskComplete').disabled=!available||taskPending||t?.active_stage!=='return_stem'||!['RUNNING','PAUSED'].includes(state)}
async function taskCommand(command,target,confirmed=false){if(taskPending||!taskState)return;if(command==='resume'&&!confirm('确认安全状态并重新授权？恢复将获取新三路图像和关节状态，重新推理，绝不续播旧动作。'))return;if(command==='complete'&&!confirm('确认整个任务已由人工验收完成，并执行完整停止和清理？'))return;const request={command,request_id:crypto.randomUUID(),expected_version:taskState.stage_version,target,confirmed:confirmed||command==='resume'};taskPending=true;renderTask(taskState);try{taskState=await api('task-command',request);$('error').textContent=''}catch(e){failure(e)}finally{taskPending=false;await refresh()}}
async function recoverTask(){let target=prompt('请输入要恢复的阶段 ID：'+stageList.map(x=>x[0]).join('、'));if(!target)return;if(!stageList.some(x=>x[0]===target)){failure(Error('未知阶段'));return}if(confirm('此操作先停止当前执行，不会立即切换硬件动作。确认人工恢复到 '+target+'？'))await taskCommand('recover',target,true)}
async function browse(target,file=false){if(busy)return;pickerTarget=target;pickerFile=file;$('chooseDirectory').hidden=file;$('browsePath').value=$(target).value||$('repository').value;$('picker').showModal();await loadDirectory()}
async function loadDirectory(){try{let d=await api('browse?path='+encodeURIComponent($('browsePath').value));$('browsePath').value=d.path;let box=$('entries');box.replaceChildren();for(let item of d.entries){let b=document.createElement('button');b.textContent=(item.directory?'📁 ':'📄 ')+item.name;b.onclick=()=>{if(item.directory){$('browsePath').value=item.path;loadDirectory()}else if(pickerFile)choosePath(item.path)};box.append(b)}}catch(e){failure(e)}}
function upDirectory(){let p=$('browsePath').value;$('browsePath').value=p.slice(0,p.lastIndexOf('/'))||'/';loadDirectory()}
function choosePath(path){$(pickerTarget).value=path;$('picker').close();invalidate();if(pickerTarget==='repository')repositoryChanged();if(['checkpoint','training_record'].includes(pickerTarget)){modelTypeChanged();$('identity').textContent='权重配置已更改，请重新核对'}}
(async()=>{let d=await api('info');token=d.token;for(let [k,v] of Object.entries(d.defaults||{}))if(fields.includes(k)){if(['record','recovery_confirmed'].includes(k))$(k).checked=v;else $(k).value=v}let savedSteps=$('steps').value;modeChanged();if(!$('mode').value.startsWith('soft-'))$('steps').value=savedSteps;for(let f of fields)$(f).addEventListener('input',invalidate);$('model_type').addEventListener('change',()=>modelTypeChanged(true));for(let f of ['checkpoint','training_record','semantics','train_rtc_max_delay','model_type','rtc_initial_delay_steps'])$(f).addEventListener('input',()=>{modelTypeChanged();$('identity').textContent='权重配置已更改，请重新核对';});$('repository').addEventListener('input',()=>{clearTimeout(repoTimer);repoTimer=setTimeout(repositoryChanged,350)});$('repository').addEventListener('change',repositoryChanged);modelTypeChanged();await repositoryChanged();showLog('execution');await refresh();setInterval(refresh,1500)})().catch(failure);
</script></html>"""


def server_for(root, *, port=8766):
    controller = DeploymentController(root)
    token = secrets.token_urlsafe(32)
    defaults = {"repository": str(root), "output": str(desktop() / "Sharpa部署记录"), "steps": 15, "rounds": 0, "task_mode": "tomato-b", "initial_stage": "pickup_1",
                "model_type": "full-delta", "mode": "rtc", "speed": "fast", "initialization": "current"}
    with suppress(OSError, ValueError):
        defaults.update(json.loads((Path(root) / ".deployment/control_last.json").read_text()))

    defaults["recovery_confirmed"] = False  # Reauthorization is never restored from a saved checkbox.
    tomato_preset = {"repository": str(root), "checkpoint": str(Path(root) / ".deployment/checkpoints/tomato-b-coarse/7999"),
                     "model_type": "full-delta", "semantics": "delta", "mode": "rtc", "speed": "fast",
                     "steps": 15, "rounds": 0, "initialization": "current", "task_mode": "tomato-b",
                     "initial_stage": "pickup_1", "recovery_confirmed": False, "training_record": "",
                     "train_rtc_max_delay": 0, "rtc_initial_delay_steps": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def send(self, status, data, kind="application/json; charset=utf-8"):
            body = data if isinstance(data, bytes) else data.encode() if isinstance(data, str) else json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if host not in allowed:
                raise ValueError("仅接受本机localhost/127.0.0.1请求")
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in allowed}:
                raise ValueError("拒绝跨来源控制请求")

        def do_GET(self):
            try:
                self.allowed()
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self.send(200, HTML, "text/html; charset=utf-8")
                elif parsed.path == "/assets/liqing-logo.png":
                    self.send(200, (Path(__file__).parent / "assets/liqing-logo.png").read_bytes(), "image/png")
                elif parsed.path == "/api/info":
                    self.send(200, {"defaults": defaults, "token": token, "tomato_preset": tomato_preset})
                elif parsed.path == "/api/status":
                    self.send(200, controller.snapshot())
                elif parsed.path == "/api/repository":
                    value = parse_qs(parsed.query).get("path", [str(root)])[0]
                    self.send(200, repository_info(value))
                elif parsed.path == "/api/browse":
                    value = parse_qs(parsed.query).get("path", [str(Path.home())])[0]
                    path = Path(value).expanduser().resolve()
                    if not path.is_dir():
                        path = path.parent
                    entries = [{"name": p.name, "path": str(p), "directory": p.is_dir()}
                               for p in path.iterdir() if not p.name.startswith(".") and (p.is_dir() or p.suffix == ".json")]
                    self.send(200, {"path": str(path), "entries": sorted(entries, key=lambda p: (not p["directory"], p["name"]))})
                else:
                    self.send(404, {"error": "Not found"})
            except Exception as error:
                self.send(400, {"error": str(error)})

        def do_POST(self):
            try:
                self.allowed()
                if self.headers.get("X-Control-Token") != token:
                    self.send(403, {"error": "控制令牌无效，请刷新页面"})
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("请求大小无效")
                data = json.loads(self.rfile.read(size))
                if self.path == "/api/inspect":
                    result = inspect_checkpoint(data["checkpoint"], training_record=data.get("training_record", ""),
                                                semantics=data.get("semantics", "auto"), cache=controller.cache,
                                                train_rtc_max_delay=data.get("train_rtc_max_delay", 0), model_type=data.get("model_type", "auto"))
                elif self.path == "/api/preview":
                    result = controller.preview(data)
                elif self.path == "/api/start-model":
                    result = controller.start_model(data["settings"])
                elif self.path == "/api/start":
                    if not data.get("approval"):
                        raise ValueError("请先核对最终配置")
                    result = controller.start(data["settings"], expected_approval=data["approval"])
                elif self.path == "/api/task-command":
                    result = controller.task_command(data)
                elif self.path == "/api/stop":
                    result = controller.stop()
                elif self.path == "/api/close-model":
                    controller.close_model()
                    result = controller.snapshot()
                else:
                    self.send(404, {"error": "Not found"})
                    return
                self.send(200, result)
            except Exception as error:
                self.send(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.controller = controller
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    server = server_for(root, port=args.port)
    closing = threading.Event()
    def shutdown(*_):
        if closing.is_set():
            return
        closing.set()
        def finish():
            if server.controller.snapshot()["model_loading"]:
                server.controller.close_model()
            else:
                server.controller.stop()
            if server.controller.thread:
                server.controller.thread.join()  # Do not kill a device disable confirmation.
            if server.controller.model is not None:
                server.controller.close_model()
            server.controller.wait_exports()
            server.shutdown()
        threading.Thread(target=finish, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Sharpa UI: http://127.0.0.1:{server.server_port} (打开页面不会启动设备)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
