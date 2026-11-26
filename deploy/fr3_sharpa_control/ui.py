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
from deploy.fr3_sharpa_runtime.config import DEFAULT_PROMPT
from .agent_settings import AGENT_PROVIDERS, AgentSettings
from deploy.fr3_sharpa_runtime.task_stages import catalog
from dataclasses import asdict

HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sharpa 部署控制台</title><style>
[hidden]{display:none!important}*{box-sizing:border-box}body{margin:0;background:#eff3f6;color:#15293e;font:14px/1.55 system-ui,sans-serif}
header{background:white;color:#123a56;border-bottom:1px solid #dae3e9;padding:6px 24px;display:flex;justify-content:space-between;align-items:center;gap:16px}
.brand{flex:1;min-width:0}.brand-logo{display:block;width:260px;max-width:100%;height:72px;background:#123a56;-webkit-mask:url('/assets/liqing-logo.png') center/contain no-repeat;mask:url('/assets/liqing-logo.png') center/contain no-repeat}

main{width:100%;padding:18px 20px 14px}.dashboard-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-bottom:16px;align-items:stretch}
.card{min-width:0;background:white;border:1px solid #dae3e9;border-radius:10px;padding:18px}h2{font-size:17px;margin:0 0 14px}label{display:block;margin-top:11px;font-weight:600}input,select,textarea{width:100%;min-width:0;padding:9px;border:1px solid #b9c8d2;border-radius:6px;background:white;font:inherit;color:inherit}
input[type=checkbox]{width:auto}.row{display:flex;gap:10px;align-items:end}.row>*{flex:1;min-width:0}.path{display:flex;gap:6px}.path input{min-width:0}.path button{flex:0 0 auto}
button{border:0;border-radius:6px;padding:9px 12px;background:#dfe9ee;color:#17394a;font:600 13px system-ui;cursor:pointer;white-space:nowrap}button.primary{background:#087b75;color:white}button.stop{background:#b63f37;color:white}button:disabled{opacity:.45;cursor:not-allowed}
.muted{color:#62778a;font-size:12px}.notice{background:#e9f5f3;padding:10px;border-radius:6px;margin:11px 0;overflow-wrap:anywhere}.error{color:#b7352f;white-space:pre-wrap;overflow-wrap:anywhere}.badge{padding:5px 11px;background:#eef3f6;color:#123a56;border-radius:20px;font-size:12px;white-space:nowrap}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,monospace;margin:8px 0;max-height:250px;overflow:auto}.control-line{padding:9px 0;border-bottom:1px solid #e5edf1}.control-line>span{margin-left:10px}.control-line .actions{margin-top:9px}.actions{display:flex;gap:8px;flex-wrap:wrap}
.agent-controls{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,2fr);gap:10px;align-items:end}.agent-controls select,.agent-controls button{min-height:42px}
@media(max-width:600px){.agent-controls{grid-template-columns:repeat(2,minmax(0,1fr))}.agent-controls>.actions{grid-column:1/-1}}
.logs .card{padding:14px;display:flex;flex-direction:column}.logs h2{font-size:15px;margin-bottom:10px}.logs pre{height:clamp(180px,21vh,270px);max-height:none;background:#132b38;color:#d5e5ed;padding:12px;border-radius:6px;margin:0;overflow:auto}.logs .is-active{border-color:#7bb5b0;box-shadow:0 0 0 1px #7bb5b0}
.page-note{margin:0;color:#62778a;font-size:12px}#directory{overflow-wrap:anywhere}#recordStatus{margin-top:6px}
#qualityDialog{width:660px}#qualityDialog::backdrop{background:rgb(19 43 56 / 40%)}#qualityDialog .grade-buttons button{flex:1;min-height:52px}dialog{border:0;border-radius:10px;width:760px;max-width:95vw}#entries{height:350px;overflow:auto}#entries button{display:block;background:white;text-align:left;width:100%;border-bottom:1px solid #e5edf1}details{margin:11px 0}summary{cursor:pointer;font-weight:600}.preview{min-height:60px}
@media(max-width:1100px){.dashboard-row{grid-template-columns:1fr}.logs pre{height:220px}header{padding:6px 18px}main{padding:14px}}
@media(max-width:520px){.brand-logo{height:60px;width:220px}.card{padding:14px}.row{flex-wrap:wrap}.row>*{flex-basis:42%}}
.dagger-controls{--dagger-column:calc((100% - 24px)/3);display:grid;grid-template-columns:repeat(2,var(--dagger-column)) calc(var(--dagger-column)/2);gap:12px;position:relative}.dagger-controls::after{content:"";position:absolute;left:calc(2*var(--dagger-column) + 24px);top:0;bottom:0;border-left:1px solid #b9c8d2;pointer-events:none}.dagger-controls button{width:100%;height:32px;white-space:normal;display:flex;align-items:center;justify-content:center;text-align:center;padding:4px}.dagger-controls button:nth-child(3n){transform:translateX(12px)}.task-stages{display:flex;gap:8px;flex-wrap:wrap}.task-stages .active{background:#087b75;color:white}.task-stages .pending{outline:2px solid #d69e36}.task-stages .passed{border-bottom:3px solid #7bb5b0}.task-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:6px}.task-layout{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-bottom:16px}.task-layout>[hidden]{display:none}@media(max-width:1100px){.task-layout{grid-template-columns:minmax(0,1fr)}}#taskError{color:#b7352f}#dagger-panel{grid-column:2}@media(max-width:1100px){#dagger-panel{grid-column:1}}#qualityEvidence{white-space:pre-line;overflow-wrap:anywhere}.grade-buttons{margin-top:10px}.grade-buttons button[aria-pressed=true]{outline:2px solid #087b75}</style><header><div class="brand"><span class="brand-logo" role="img" aria-label="厘清智能"></span></div><span id="phase" class="badge">空闲</span></header>
<main><section class="dashboard-row configuration" aria-label="部署配置与控制"><article class="card" id="weights-card"><h2>1 · 权重与代码</h2>
<label>部署仓库目录</label><div class="path"><input id="repository"><button onclick="browse('repository')">选择目录</button></div>
<div class="muted" id="repoInfo">选择本机部署目录，核对当前代码与权重配置。</div>
<label>Checkpoint 步骤目录</label><div class="path"><input id="checkpoint" placeholder="直接包含 params/ 和 assets/ 的目录"><button onclick="browse('checkpoint')">选择目录</button></div>
<div id="unifiedPromptFields"><label for="prompt">训练时的统一英文 Prompt</label><textarea id="prompt" rows="3" placeholder="填写该权重训练时的统一任务指令"></textarea></div>
<label for="image_input">图像输入（独立于任务流程）</label><select id="image_input"><option value="auto">自动识别图像路数</option><option value="three">三路 · 原始全图</option><option value="four">四路 · 原始三路＋局部图</option></select>
<p class="muted">放回盘中调整和手中调整都可有三路／四路版本。输入选项必须与实际权重匹配；新训练的三路对照完成核验后再接入。</p>
<label>模型类型（手动选择）</label><select id="model_type"><option value="full">Full Absolute · 全量绝对位置</option><option value="full-delta">Full Delta · 双臂增量</option><option value="lora">LoRA Absolute</option><option value="paper-delta">Paper Delta</option><option value="train-rtc-absolute">Train RTC Absolute</option><option value="train-rtc-delta">Train RTC Delta</option><option value="train-paper-rtc">Train Paper RTC</option><option value="auto">自动识别（未知项可手动选择）</option></select>
<div class="muted">直接选择checkpoint及其真实训练类型，无需生成、复制部署契约。仍会检查权重结构和配套统计；Paper使用原有算法资产。</div>
<details><summary>训练配置（可选，辅助自动识别）</summary><label>归档训练 JSON（可选）</label><div class="path"><input id="training_record"><button onclick="browse('training_record',true)">选择文件</button></div></details>
<details id="trainRTCSettings"><summary>Train RTC 专用设置</summary><label>训练 RTC 最大延迟（0 自动识别）</label><input id="train_rtc_max_delay" type="number" min="0" max="49" value="0"><div class="muted">手动选择Train RTC时必填实际训练上限（例如10）；普通RTC无需填写。没有训练JSON也可直接手动配置运行。</div>
<label>Train RTC 初始延迟（0 使用训练上限，最多25）</label><input id="rtc_initial_delay_steps" type="number" min="0" max="25" value="0">
</details><div class="actions" style="margin-top:14px"><button onclick="inspect()">自动识别并填入</button></div>
<div id="identity" class="notice" style="white-space:pre-line">覆盖：30Hz／58D 的全量 Absolute、普通 Delta、Paper Delta、Train RTC、Train Paper RTC，以及 Absolute LoRA。Paper按资产识别；LoRA Delta与其他结构尚不支持。实际结构在GPU加载时再次核对。</div>
</article><article class="card" id="execution-card"><h2>2 · 本次执行配置</h2>
<div class="row"><div><label>执行模式</label><select id="mode" onchange="modeChanged()"><option value="serial">普通串行</option><option value="rtc">RTC</option><option value="train-rtc">Train RTC · 训练前缀条件</option><option value="train-paper-rtc">Train Paper RTC · 训练前缀＋Paper</option><option value="paper-rtc">Paper RTC · 时间步归一化</option></select></div><div><label>启动姿态</label><select id="initialization"><option value="current">Current · 从当前状态推理</option><option value="home">Home · 先回固定位置再推理</option></select></div></div>
<div class="row"><div><label>速度</label><select id="speed"><option value="slow">Slow · 0.1倍</option><option value="medium">Medium · 0.5倍</option><option value="fast">Fast · 模型原速</option></select></div><div><label id="stepsLabel">每轮执行的模型动作数</label><input id="steps" type="number" value="15" min="1" max="50"></div><div><label>轮数（0为持续）</label><input id="rounds" type="number" value="0" min="0"></div></div>
<div id="modeHint" class="notice"></div>
<label for="task_mode">部署方式</label><select id="task_mode"><option value="none">统一 Prompt · 不分阶段</option><option value="tomato-b">分阶段 · 放回盘中调整（七阶段）</option><option value="tomato-in-hand119">分阶段 · 手中调整（五阶段）</option></select>
<p class="muted">摘一颗，或统一指令摘一串：选不分阶段。只有训练时使用不同阶段指令的权重才选分阶段；Agent 可选。</p>
<label><input id="record" type="checkbox"> 录制 ROSbag及视频（四路模型额外保存ROI与四联视频）</label>
<div id="cropFields"><label>四路模型局部图定位器目录（model.pt＋receipt.json）</label><input id="crop_locator" placeholder="留空使用所选仓库/.deployment/crop_locator"></div>
<div id="stageRecovery"><label><input type="checkbox" id="recovery_confirmed"> 人工恢复：已停止旧执行、检查设备，明确授权从选定阶段重新开始</label>
<div class="muted">非首次阶段必须勾选并重新核对配置。恢复会获取新观测和首块，不复用旧动作。</div></div>
<label>归档父目录</label><div class="path"><input id="output"><button onclick="browse('output')">选择目录</button></div><div class="muted">自动创建 日期/小时.分钟（如15.43）；同一分钟重启自动加后缀，不覆盖旧记录。</div>
</article><article class="card" id="control-card"><h2>3 · 推理与执行</h2>
<p class="muted">先启动推理，预热就绪后再核对并启动执行。番茄任务执行端就绪后，还需点击下方“开始当前阶段”。执行自动完成硬件 up；停止执行等效一次 Ctrl-C，等待失能确认及硬件清理，推理端保持运行。停止后可重新选择 Current/Home，再次执行自动 up。</p><p class="muted">部署页面获得焦点时，B 踏板让空闲双臂回 Home；清理期间踩 B 会排队等待清理完成，执行期间 B 无效。编辑输入框时踏板无效。</p><div id="message" class="notice">尚未启动设备</div><div id="error" class="error"></div><div class="control-line"><strong>推理端</strong><span id="modelState" class="muted">未启动</span><div class="actions"><button id="startModel" class="primary" onclick="startModel()">启动推理</button><button id="closeModel" class="stop" onclick="closeModel()" disabled>停止推理</button></div></div>
<div class="control-line"><strong>执行端</strong><span id="executionState" class="muted">未运行</span><div class="actions"><button id="preview" onclick="preview()">核对执行配置</button><button id="start" class="primary" onclick="start()" disabled>启动执行</button><button id="stop" class="stop" onclick="stop()" disabled>停止执行</button></div></div>
<details id="planDetails"><summary id="previewSummary">执行配置明细</summary><pre id="previewText" class="preview">请选择配置后点击“核对执行配置”。</pre></details><div id="directory" class="muted"></div><div id="recordStatus" class="muted"></div>
<div id="cropStatus" class="notice">局部图：等待四路模型启动及新观测</div><img id="cropPreview" style="display:none;width:100%;max-width:672px" alt="当前原图框与第四路模型输入">

</article></section>
<div class="task-layout" id="taskLayout">
<section class="card" id="taskPanel" aria-label="番茄采摘任务状态机"><h2>番茄采摘任务状态机</h2>
<p class="muted" id="taskInstructions">执行端就绪后，点击「开始当前阶段」。</p>
<div class="agent-controls">
<div><label for="agentProvider">Agent 模型</label><select id="agentProvider" onchange="agentProviderChanged()"><option value="mengxiang">mengxiang</option><option value="tairan">tairan · 四路</option></select></div>
<div class="actions"><button id="startAgent" class="primary" onclick="agentCommand('start-agent')">启动 Agent</button><button id="stopAgent" class="stop" onclick="agentCommand('stop-agent')" disabled>停止 Agent</button></div>
</div>
<p id="agentStatus" class="muted" role="status">Agent 未启动</p><div id="agentError" class="error"></div>
<details id="agentPaths"><summary>首次配置／修改工作机上的 Agent 路径</summary>
<p class="muted">选择 Agent 会自动填入对应配置。路径不符时在这里修改；无需选择运行环境。</p>
<div class="row"><div><label>Agent 仓库目录</label><input id="agentRepository"></div><div><label>Agent 权重路径（可相对 Agent 仓库）</label><input id="agentCheckpoint"></div></div>
<div class="row"><div><label>Agent Python 路径</label><input id="agentPython"></div><div><label>编码器目录</label><input id="agentPretrained"></div></div>
<div class="row"><div><label>工作机 Gello 仓库</label><input id="agentGello"></div><div><label>ROS 消息 overlay（可相对部署仓库）</label><input id="agentOverlay"></div></div>
<div class="actions" style="margin-top:12px"><button id="checkAgent" onclick="checkAgentEnvironment()">检查工作机配置</button></div><div id="agentEnvironmentReport" role="status"></div></details>
<div id="agentProbabilities" class="task-grid"></div><details><summary>查看 Agent 预测与切换规则</summary><p class="muted">Agent 启动并就绪后，按已有概率阈值请求下一阶段。人工可随时点击任意阶段，旧预测随阶段版本失效。停止 Agent 后由人工切阶段。最终阶段结束后点击停止执行。</p></details>
<div id="taskStages" class="task-stages"></div>
<div id="autoActivePrompt" class="notice" hidden><strong id="autoPromptStage"></strong><p id="autoPromptText"></p></div>
<input type="hidden" id="initial_stage">
<details><summary>当前实际生效的英文 Prompt</summary><pre id="taskPrompt">尚未接入阶段动作</pre></details>
<div class="actions"><button id="taskStart" class="primary" onclick="taskCommand(taskState?.execution_state==='PAUSED'?'resume':'start')" disabled>开始当前阶段</button><button id="taskPause" title="暂停当前动作，执行端确认可恢复后点击继续当前阶段" onclick="taskCommand('pause')" disabled>暂停</button><button class="stop" onclick="stop()">停止执行</button></div>
<p class="notice">手中调整任务切换会等当前动作块发完，再用新观测和新 Prompt 平滑接入下一阶段。其他任务沿用原在线交接。超时、越界或反馈异常会停止并清理。暂停仍需实际安全确认；当前缺少双臂静止保持 ACK，暂停后自动恢复受阻止。关闭浏览器页面不会停止任务。请点击“停止执行”结束任务。本机控制服务失联时会停止任务。这些按钮属于软件控制，不是硬件急停。</p><div id="taskError" role="alert"></div></section>
<section class="card" id="dagger-panel" aria-label="DAgger 与录制质量"><h2>DAgger · 接管与数据</h2>
<label><input id="dagger" type="checkbox" onchange="daggerChanged()"> 本次启用 DAgger 接管与自动标注（同时开启录制）</label>
<details><summary>DAgger 数据保存设置</summary>
<label for="dagger_output">保存父目录</label><div class="path"><input id="dagger_output" oninput="saveControlPreferences({dagger_output:this.value})" placeholder="留空使用归档父目录下的 DAgger"><button onclick="browse('dagger_output')">浏览…</button></div>
<div class="notice">A 优等 → A/　B 一般 → B/　C 报错 → C/　D 放弃 → D/<br>未评级暂存 pending/；异常退出后弹窗建议选择 C。评级后等待后台导出结束，将 rosbag 移至对应等级/episodeXX。部署日志另存原日期目录。</div>
</details>
<p class="muted">左臂＋左手由人接管，右侧保持。同一控制服务持续运行；恢复策略会重新观测与推理。</p>
<div id="daggerStatus" class="notice" role="status">未启用 DAgger</div>
<div class="dagger-controls" aria-label="DAgger 操作台">
<button id="daggerTakeover" onclick="daggerCommand('takeover')" disabled>接管左侧</button>
<button id="daggerTakeoverRight" disabled title="暂未实现">接管右侧</button>
<button id="daggerResume" class="primary" onclick="daggerCommand('resume')" disabled>恢复策略</button>
<button id="daggerFollow" title="A 踏板切换左侧跟随／保持" onclick="daggerCommand(daggerState?.mode==='HUMAN_LEFT'?'hold':'follow')" disabled>左侧开始跟随</button>
<button id="daggerFollowRight" disabled title="C 踏板预留，暂未实现">右侧开始跟随</button>
</div>
<p class="muted">Space 踏板切换左侧接管／恢复策略；L 踏板开始当前阶段，再踩一次停止执行。A 踏板切换左侧跟随／保持，C 踏板为右侧跟随预留。恢复策略前自动停止跟随并等待保持确认。请让部署页面获得焦点，编辑输入框或弹窗打开时踏板停用，长踩只触发一次。</p><div id="daggerError" class="error" role="alert"></div>
<label for="qualityNote">最近一条录制的批注（可选）</label><textarea id="qualityNote" rows="2" maxlength="2000" placeholder="填写最近一次录制的批注"></textarea>
<div class="actions" style="margin-top:10px"><button id="submitQualityNote" onclick="submitQualityNote()" disabled>提交批注</button></div>
<p id="qualityNoteStatus" class="muted" role="status" aria-live="polite"></p>
<p id="qualityEvidence" class="notice" role="status">等待录制收尾。</p>
<p class="muted">评级独立于执行结果和录制完整性。DAgger 按评级归档 rosbag 到等级/episodeXX，D 保留数据；评级不表示已通过训练数据校验。后台导出完成后移动。</p><div id="qualityError" class="error" role="alert"></div>
</section>
</div>
<section class="dashboard-row logs" id="logs" aria-label="三个终端日志">
<article class="card" id="terminal-hardware"><h2>终端一 · 硬件日志</h2><pre id="log-hardware" role="log" aria-label="硬件日志">尚无硬件日志</pre></article>
<article class="card" id="terminal-inference"><h2>终端二 · 推理日志</h2><pre id="log-inference" role="log" aria-label="推理日志">尚无推理日志</pre></article>
<article class="card" id="terminal-execution"><h2>终端三 · 执行日志</h2><pre id="log-execution" role="log" aria-label="执行日志">尚无执行日志</pre></article>
</section>
<p class="page-note">打开页面不会连接设备；普通部署关闭页面不等于停止执行；任务心跳由本机控制服务维持，关闭页面不会停止任务。请优先使用停止按钮。完整日志保存在归档目录。</p>
</main>
<dialog id="qualityDialog" aria-labelledby="qualityTitle">
<h2 id="qualityTitle">数据质量评价</h2>
<p id="qualityEpisode" class="notice"></p><p id="qualityResult"></p>
<p>请选择本次数据质量，点击后保存。</p>
<div class="actions grade-buttons">
<button id="gradeA" onclick="rateRecording('A')">A · 优等</button>
<button id="gradeB" onclick="rateRecording('B')">B · 一般</button>
<button id="gradeC" onclick="rateRecording('C')">C · 报错</button>
<button id="gradeD" onclick="rateRecording('D')">D · 放弃</button>
</div><p id="qualityDialogError" class="error" role="alert"></p>
<div class="actions"><button onclick="closeQualityDialog()">稍后评价</button></div>
</dialog>
<dialog id="picker"><h2>选择工作机上的路径</h2><div class="path"><input id="browsePath"><button onclick="loadDirectory()">打开</button><button onclick="upDirectory()">上一级</button></div><div id="entries"></div><div class="actions"><button class="primary" id="chooseDirectory" onclick="choosePath(document.getElementById('browsePath').value)">选择此目录</button><button onclick="document.getElementById('picker').close()">取消</button></div></dialog>
<script>
let token='', approved=null, approvedPlan=null, approvedModelKey=null, busy=false, executionActive=false, executionStarting=false, pickerTarget='', pickerFile=false, repoModes=['serial','rtc'], modelModes=null, previousMode='serial', repoTimer=null;
const $=id=>document.getElementById(id);
const fields=['repository','checkpoint','mode','initialization','speed','steps','rounds','record','output','training_record','train_rtc_max_delay','model_type','rtc_initial_delay_steps','task_mode','initial_stage','recovery_confirmed','crop_locator','prompt','image_input','dagger','dagger_output'];
function config(){let c={};for(let f of fields)c[f]=['record','recovery_confirmed','dagger'].includes(f)?$(f).checked:['steps','rounds','train_rtc_max_delay','rtc_initial_delay_steps'].includes(f)?Number($(f).value):$(f).value;return c}
function daggerChanged(){if($('dagger').checked)$('record').checked=true;invalidate()}
let daggerState=null,daggerPending=false;
function renderDagger(d){
 daggerState=d;const mode=d?.mode,valid=executionActive&&!daggerPending&&!d?.communication_error;
 const labels={STARTING:'准备中',IDLE:'保持，等待开始阶段',POLICY:'策略执行',HOLDING:'等待实测保持确认',HOLD:'保持已确认',HUMAN_LEFT:'左臂＋左手人工跟随，右侧保持',RESUMING:'等待新观测和新策略',STOPPED:'已结束'};
 $('daggerStatus').textContent=d?[labels[mode]||mode,d.intervention_id?'接管编号 '+d.intervention_id:'',d.input_ready?'左侧输入就绪':d.input_error||'',d.reason||''].filter(Boolean).join(' · '):'未启用 DAgger';
 $('daggerTakeover').disabled=!valid||!['POLICY','RESUMING'].includes(mode);
 $('daggerFollow').disabled=!valid||!(mode==='HUMAN_LEFT'||mode==='HOLD'&&d?.input_ready);
 $('daggerFollow').textContent=mode==='HUMAN_LEFT'?'左侧保持':'左侧开始跟随';
 $('daggerResume').disabled=!valid||!['HOLD','HOLDING','HUMAN_LEFT'].includes(mode);
 if(d?.communication_error)$('daggerError').textContent=d.communication_error;
}
async function daggerCommand(command){
 if(!daggerState||daggerPending)return;
 let d=daggerState;const runId=d.run_id;daggerPending=true;renderDagger(d);
 const send=async action=>{
  if(d.run_id!==runId)throw Error('DAgger 采集身份已变化，请重新操作');
  const result=await api('dagger-command',{command:action,run_id:runId,request_id:crypto.randomUUID(),expected_generation:d.generation});
  if(result.run_id!==runId)throw Error('DAgger 采集身份已变化');
  d=result;renderDagger(d);
 };
 try{
  $('daggerError').textContent='';
  if(command==='resume'){
   if(d.mode==='HUMAN_LEFT')await send('hold');
   const deadline=Date.now()+6000;
   while(d.mode==='HOLDING'){
    if(Date.now()>=deadline)throw Error('保持确认超时，请检查状态后重新恢复策略');
    await new Promise(resolve=>setTimeout(resolve,100));
    const status=await api('status');
    if(!status.execution_running||status.dagger?.run_id!==runId||status.dagger?.communication_error)
     throw Error('执行已结束或 DAgger 通信失效，恢复已取消');
    d=status.dagger;renderDagger(d);
   }
   if(d.mode!=='HOLD')throw Error('当前状态无法恢复策略，请重新操作');
  }
  await send(command);await refresh();
 }catch(e){$('daggerError').textContent=e.message}
 finally{daggerPending=false;renderDagger(daggerState)}
}
let daggerPedalDown=false;
function daggerPedalKey(event){
 if(event.code!=='Space')return;
 if(event.type==='keyup'){daggerPedalDown=false;return}
 const target=event.target;
 if(event.isComposing||event.ctrlKey||event.altKey||event.metaKey||event.shiftKey||
  target?.isContentEditable||target?.closest?.('input,textarea,select,[role="textbox"]')||document.querySelector('dialog[open]'))return;
 if(!executionActive||!daggerState)return;
 event.preventDefault();
 if(event.repeat||daggerPedalDown)return;
 daggerPedalDown=true;
 if(daggerPending||daggerState.communication_error)return;
 const command=['POLICY','RESUMING'].includes(daggerState.mode)?'takeover':
  ['HOLD','HOLDING','HUMAN_LEFT'].includes(daggerState.mode)?'resume':null;
 if(!command||$(command==='takeover'?'daggerTakeover':'daggerResume').disabled)return;
 daggerCommand(command);
}
document.addEventListener('keydown',daggerPedalKey);
document.addEventListener('keyup',daggerPedalKey);
const followPedalsDown=new Set();
function followPedalKey(event){
 if(!['KeyA','KeyC'].includes(event.code))return;
 if(event.type==='keyup'){followPedalsDown.delete(event.code);return}
 const target=event.target;
 if(event.isComposing||event.ctrlKey||event.altKey||event.metaKey||event.shiftKey||
  target?.isContentEditable||target?.closest?.('input,textarea,select,[role="textbox"]')||document.querySelector('dialog[open]'))return;
 if(!executionActive||!daggerState)return;
 event.preventDefault();
 if(event.repeat||followPedalsDown.has(event.code))return;
 followPedalsDown.add(event.code);
 if(event.code==='KeyC'||daggerPending||daggerState.communication_error)return;
 const command=daggerState.mode==='HUMAN_LEFT'?'hold':daggerState.mode==='HOLD'?'follow':null;
 if(!command||$('daggerFollow').disabled)return;
 daggerCommand(command);
}
document.addEventListener('keydown',followPedalKey);
document.addEventListener('keyup',followPedalKey);
let taskPedalDown=false,taskPedalPending=false,taskPedalStartedRun=null;
async function taskPedalKey(event){
 if(event.code!=='KeyL')return;
 if(event.type==='keyup'){taskPedalDown=false;return}
 if(event.repeat||taskPedalDown)return;
 taskPedalDown=true;
 if(event.isComposing||event.ctrlKey||event.altKey||event.metaKey||event.shiftKey)return;
 const target=event.target;
 if(target?.isContentEditable||target?.closest?.('input,textarea,select,[role="textbox"]')||document.querySelector('dialog[open]'))return;
 if(taskPedalPending||!executionActive||!taskState||$('task_mode').value==='none')return;
 const starting=['READY','PAUSED'].includes(taskState.execution_state)&&taskPedalStartedRun!==taskState.run_id;
 if(starting?$('taskStart').disabled:$('stop').disabled)return;
 event.preventDefault();taskPedalPending=true;
 const runId=taskState.run_id;
 try{if(starting){if(await taskCommand(taskState.execution_state==='PAUSED'?'resume':'start'))taskPedalStartedRun=runId}else await stop()}
 finally{taskPedalPending=false}
}
document.addEventListener('keydown',taskPedalKey);
document.addEventListener('keyup',taskPedalKey);
window.addEventListener('blur',()=>{daggerPedalDown=false;taskPedalDown=false;followPedalsDown.clear()});
let homePedalDown=false,homeRequestPending=false,homeRunning=false;
function homePedalKey(event){
 if(event.code!=='KeyB')return;
 if(event.type==='keyup'){homePedalDown=false;return}
 if(event.repeat||homePedalDown)return;
 homePedalDown=true;
 if(event.isComposing||event.ctrlKey||event.altKey||event.metaKey||event.shiftKey)return;
 const target=event.target;
 if(target?.isContentEditable||target?.closest?.('input,textarea,select,[role="textbox"]')||document.querySelector('dialog[open]'))return;
 if(executionStarting||homeRunning||homeRequestPending)return;
 event.preventDefault();requestArmsHome();
}
async function requestArmsHome(){
 homeRequestPending=true;
 try{const d=await api('home-arms',{repository:$('repository').value});if(d.home)homeRunning=d.home.running;await refresh()}
 catch(e){failure(e)}finally{homeRequestPending=false}
}
document.addEventListener('keydown',homePedalKey);
document.addEventListener('keyup',homePedalKey);
window.addEventListener('blur',()=>{homePedalDown=false});
async function api(path,data){const r=await fetch('/api/'+path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':token},body:JSON.stringify(data)});let d=await r.json();if(!r.ok)throw Error(d.error);return d}
function failure(e){$('error').textContent=e.message}
let preferenceSaves=Promise.resolve();
function saveControlPreferences(values){
 preferenceSaves=preferenceSaves.catch(()=>{}).then(()=>api('preferences',values));
 preferenceSaves.catch(failure);
 return preferenceSaves;
}
function restoreControlPreferences(values){
 if(typeof values?.dagger_output==='string')$('dagger_output').value=values.dagger_output;

}
function invalidate(){approved=null;approvedPlan=null;approvedModelKey=null;$('previewSummary').textContent='执行配置明细';$('start').disabled=true;$('error').textContent=''}
function modeChanged(){let mode=$('mode').value;$('steps').disabled=false;$('steps').value=15;$('steps').max=['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(mode)?24:50;$('stepsLabel').textContent=['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(mode)?'重规划触发源节点':'每轮执行的模型动作数';$('speed').disabled=['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(mode);if(['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(mode))$('speed').value='fast';else if(['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(previousMode))$('speed').value='slow';previousMode=mode;$('modeHint').textContent=['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes(mode)?'原速后台重规划；此处节点数不是串行的固定执行前缀。':'每轮预测50，执行填写的前N个动作，然后重新观测推理。';invalidate()}
function modeNames(modes){return modes.map(m=>Array.from($('mode').options).find(o=>o.value===m).textContent).join('、')}
function updateModes(){let modes=repoModes.filter(m=>!modelModes||modelModes.includes(m));for(let o of $('mode').options)o.disabled=!modes.includes(o.value);if(!modes.includes($('mode').value)){$('mode').value=modes[0]||'serial';modeChanged()}if(modelModes)$('identity').textContent=$('identity').textContent.replace(/可用模式：.*/, '可用模式：'+modeNames(modes));return modes}
function modelTypeChanged(reset=false){let t=$('model_type').value,train=t.startsWith('train-');$('trainRTCSettings').open=train;modelModes=t==='auto'?null:t==='lora'?['serial']:train?['serial',t==='train-paper-rtc'?'train-paper-rtc':'train-rtc']:['serial',t==='paper-delta'?'paper-rtc':'rtc'];if(reset&&!train&&t!=='auto'){$('train_rtc_max_delay').value=0;$('rtc_initial_delay_steps').value=0}updateModes();invalidate()}
function modelContractText(model){
 const m=model.tomato_model;
 if(!m)return '图像输入：'+(model.input_views?model.input_views+'路':'待手动选择');
 return `任务：${m.task_name} · 输入：${m.input_name}\n训练版本：${m.dataset_version||"自定义配置"} · checkpoint ${m.checkpoint_step||"当前选择"}`;
}
let inspectedViews=null;
function updateDeploymentFields(){
 const staged=$('task_mode').value!=='none';
 $('unifiedPromptFields').hidden=staged;
 $('stageRecovery').hidden=!staged;
 $('taskPanel').hidden=!staged;
 $('taskLayout').hidden=false;
 $('cropFields').hidden=$('image_input').value==='three'||($('image_input').value==='auto'&&inspectedViews===3);
 $('cropStatus').hidden=$('cropFields').hidden;
 $('cropPreview').hidden=$('cropFields').hidden;
}
async function inspect(){try{
 const c=config();const d=await api('inspect',{...c,model_type:'auto',semantics:'auto',train_rtc_max_delay:0,image_input:'auto'});
 if(d.profile){$('model_type').value=d.train_rtc?(d.paper_algorithm?'train-paper-rtc':d.semantics==='delta'?'train-rtc-delta':'train-rtc-absolute'):d.paper_algorithm?'paper-delta':d.profile;}
 $('train_rtc_max_delay').value=d.train_rtc?d.train_rtc.max_delay:0;
 if(d.suggestions?.prompt)$('prompt').value=d.suggestions.prompt;
 if(d.input_views)$('image_input').value=d.input_views===4?'four':'three';
 if(d.suggestions?.task_mode){$('task_mode').value=d.suggestions.task_mode;taskModeChanged();}
 modelTypeChanged();modelModes=d.supported_modes;inspectedViews=d.input_views;
 const preferred=d.supported_modes.at(-1);if(d.profile&&preferred){$('mode').value=preferred;modeChanged();}
 updateDeploymentFields();const modes=updateModes();
 $('identity').textContent=`${modelContractText(d)}\n${d.finetune_mode} · ${d.semantics||'动作语义待选'} · 30 Hz / 58D / horizon 50\n${d.evidence}${d.train_rtc?' · Train RTC 最大延迟='+d.train_rtc.max_delay+'步':''}\n归一化：${d.normalization}\n${(d.warnings||[]).join('\\n')}\n已识别项已填入；其余设置可手动修改。可用模式：${modeNames(modes)}`;invalidate();
 }catch(e){failure(e)}}
async function repositoryChanged(){try{let d=await api('repository?path='+encodeURIComponent($('repository').value));repoModes=d.modes;$('repoInfo').textContent='当前分支：'+d.branch+'；仅可选当前代码支持的模式';updateModes();invalidate()}catch(e){failure(e)}}
async function preview(){try{const c=config();const p=await api('preview',c);$('previewText').textContent=JSON.stringify({分支:p.branch,提交:p.commit.slice(0,8),权重:p.model.checkpoint,模型:p.model.profile,证据:p.model.evidence,归一化:p.model.normalization,论文设置:p.model.paper_algorithm?{归一化:p.model.paper_algorithm.mode,相关噪声系数:p.model.paper_algorithm.beta}:null,方式:p.description,任务模式:c.task_mode,图像输入:p.model.input_views+'路',统一Prompt:c.task_mode==='none'?c.prompt:'按阶段指令',初始阶段:c.initial_stage,人工恢复确认:c.recovery_confirmed,姿态:c.initialization,速度:c.speed,轮数:c.rounds,录制:c.record,归档:p.settings.output},null,2);approved=JSON.stringify(c);approvedPlan=p.approval;approvedModelKey=JSON.stringify([p.settings.repository,p.commit,p.model.identity,p.model.profile,p.port,p.settings.mode]);$('previewSummary').textContent='执行配置已核对 · 点击展开';$('error').textContent='';await refresh()}catch(e){invalidate();failure(e)}}
async function startModel(){try{const c=config();showLog('inference');await api('start-model',{settings:c});await refresh()}catch(e){failure(e)}}
function showLog(name){for(let key of ['hardware','inference','execution'])$('terminal-'+key).classList.toggle('is-active',key===name)}
async function start(){
if(executionStarting)return;
try{
 if(approved!==JSON.stringify(config()))throw Error('配置已改变，请重新核对');
 executionStarting=true;$('start').disabled=true;renderTask(taskState);
 const status=await api('start',{settings:config(),approval:approvedPlan});
 executionActive=status.execution_running;busy=status.running;showLog('execution');
 await refresh();
}catch(e){failure(e)}finally{executionStarting=false;renderTask(taskState)}
}
async function stop(){queuedManualStage=null;try{await api('stop',{});await refresh()}catch(e){failure(e)}}
async function closeModel(){try{await api('close-model',{});await refresh()}catch(e){failure(e)}}
async function refresh(){try{let s=await api('status');busy=s.running;
executionActive=s.execution_running;
homeRunning=!!s.home?.running;
renderDagger(s.dagger);
renderAgent(s.agent);renderTask(s.task);
$('phase').textContent=({idle:'空闲',validating:'配置检查',hardware_up:'启动硬件',model_loading:'推理预热中',model_stopping:'停止推理中',model_ready:'推理就绪',model_off:'推理已停止',recording_ready:'准备录制',running:'执行中',stopping:'停止执行中',failed:'失败',stopped:'执行已停止',finished:'执行已完成'})[s.phase]||s.phase;
$('message').textContent=s.message;$('modelState').textContent=s.model_loading?'正在加载／停止':s.model_ready?(s.model_owned?'已就绪':'已就绪（外部终端）'):s.model_running?'未就绪':'未启动';$('executionState').textContent=s.execution_running?'执行／清理中':'未运行';
if(s.interaction_crop){let r=s.interaction_crop;$('cropStatus').textContent=`局部图：${r.mode} · ${r.source_monotonic_ns==null?'模型预热样本':r.age_s>2?'画面已过期':'最新查询'} · 检测 ${r.detector_ms.toFixed(1)} ms / 恢复 ${(r.recovery_ms??0).toFixed(1)} ms / 裁剪合计 ${r.crop_total_ms.toFixed(1)} ms · 原图 ${r.source_size.join('×')} · ${JSON.stringify(r.counts)}`;$('cropPreview').style.display='block';$('cropPreview').src='/api/crop-preview?t='+r.updated_wall_time}else{$('cropPreview').style.display='none';$('cropStatus').textContent='局部图：等待四路模型启动及新观测'}
$('stop').disabled=homeRunning?s.home.phase==='stopping':!s.execution_running||s.stop_requested;$('stop').textContent=homeRunning?'取消回Home':'停止执行';$('start').disabled=busy||homeRunning||executionStarting||!s.model_ready||JSON.stringify(s.model_key)!==approvedModelKey||approved!==JSON.stringify(config());$('startModel').disabled=busy||!$('checkpoint').value;$('closeModel').disabled=s.execution_running||(!s.model_owned&&!s.model_loading);$('preview').disabled=busy;
$('directory').textContent=s.directory?'本次日志目录：'+s.directory:'';$('recordStatus').textContent=[s['recording.json']?'录制：'+s['recording.json'].status:'',s['video_export.json']?'视频：'+s['video_export.json'].status:'',s.exports.length?'后台导出：'+s.exports.length+'项':''].filter(Boolean).join(' · ');
refreshRecordings();
for(let key of ['hardware','inference','execution']){let pre=$('log-'+key),end=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-8;let text=s.logs[key]||({'hardware':'尚无硬件日志','inference':'尚无推理日志','execution':'尚无执行日志'})[key];if(pre.textContent!==text){pre.textContent=text;if(end)pre.scrollTop=pre.scrollHeight}}
for(let f of fields)$(f).disabled=busy;if(!busy){$('steps').disabled=false;$('speed').disabled=['rtc','paper-rtc','train-rtc','train-paper-rtc'].includes($('mode').value)}}catch(e){failure(e)}}

let qualityRows=[],qualityOutput='',qualityBusy=false,qualityLoading=false,qualityRefreshAt=0;
let qualityReview=null;
const qualityDismissed=new Set();
function selectedRecording(){return qualityRows.find(r=>r.review_ready)}
function closeQualityDialog(){
 if(qualityBusy)return;
 if(qualityReview)qualityDismissed.add(qualityReview.row.run_id);
 $('qualityDialog').close();qualityReview=null;
}
function showQualityDialog(row,output=qualityOutput){
 if(qualityReview||!row?.can_rate)return;
 qualityReview={row:structuredClone(row),output};
 $('qualityEpisode').textContent=row.archive?.episode||row.relative;
 const failed=row.execution.outcome==='FAILED'||row.recording_status!=='COMPLETE';
 $('qualityResult').textContent=failed?'本次采集异常退出，建议选择 C · 报错。 '+(row.execution.reason||row.recording_error||''):
  '本次采集已正常结束，请按实际数据质量评价。';
 $('qualityDialogError').textContent='';
 for(const g of ['A','B','C','D'])$('grade'+g).disabled=false;
 $('gradeC').classList.toggle('primary',failed);
 $('qualityDialog').showModal();
}
function renderRecordingQuality(){
 const r=selectedRecording();
 $('qualityNote').disabled=qualityBusy||!r?.can_rate;
 $('submitQualityNote').disabled=qualityBusy||!r?.can_rate||!r?.grade||!$('qualityNote').value.trim();
 $('submitQualityNote').textContent=qualityBusy?'保存中…':'提交批注';
 $('qualityEvidence').textContent=r?[
  r.archive_directory||r.directory,
  r.archive_pending?'等待后台导出完成后归档到 '+r.grade+'/':'',
  '执行：'+r.execution.outcome+' · 录制：'+r.recording_status+' · 评级：'+(r.grade?r.grade+' '+r.label:'待评价'),
  r.note?'已保存批注：'+r.note:'',
  r.execution.reason||'',r.recording_error||'',r.archive_error?'归档失败，原数据保留：'+r.archive_error:'',...r.warnings
 ].filter(Boolean).join(String.fromCharCode(10)):'暂无已收尾的录制。';
}
async function refreshRecordings(force=false){
 if(qualityLoading||qualityBusy||(!force&&Date.now()<qualityRefreshAt))return;
 const output=$('output').value;if(!output)return;
 qualityLoading=true;qualityRefreshAt=Date.now()+1500;
 try{
  const rows=await api('recordings',{output});if(output!==$('output').value)return;
  rows.sort((a,b)=>(b.execution.ended_unix_ns||0)-(a.execution.ended_unix_ns||0));
  const previous=selectedRecording(),changedRoot=output!==qualityOutput;
  qualityRows=rows;qualityOutput=output;
  const current=selectedRecording();
  if(changedRoot||previous?.run_id!==current?.run_id){
   $('qualityNote').value='';$('qualityNoteStatus').textContent='';
  }
  renderRecordingQuality();
  const pending=rows.find(r=>r.archive&&r.can_rate&&!r.history.some(e=>!e.automatic)&&!qualityDismissed.has(r.run_id));
  if(!executionActive&&pending)showQualityDialog(pending,output);
 }catch(e){$('qualityError').textContent=e.message}finally{qualityLoading=false}
}
async function rateRecording(grade){
 const review=qualityReview;if(qualityBusy||!review)return;
 const r=review.row;qualityBusy=true;
 for(const g of ['A','B','C','D'])$('grade'+g).disabled=true;
 try{
  const note=selectedRecording()?.run_id===r.run_id?($('qualityNote').value.trim()||r.note):r.note;
  await api('rate-recording',{output:review.output,relative:r.relative,run_id:r.run_id,expected_revision:r.revision,grade,note});
  qualityBusy=false;closeQualityDialog();
 }catch(e){$('qualityDialogError').textContent=e.message}
 finally{
  qualityBusy=false;
  for(const g of ['A','B','C','D'])$('grade'+g).disabled=false;
  await refreshRecordings(true);renderRecordingQuality();
 }
}
async function submitQualityNote(){
 const r=selectedRecording();if(qualityBusy||!r?.can_rate||!r?.grade||!$('qualityNote').value.trim())return;
 const output=qualityOutput,note=$('qualityNote').value;
 const feedback=$('qualityNoteStatus');
 qualityBusy=true;renderRecordingQuality();
 feedback.className='muted';feedback.textContent='正在保存批注…';
 try{
  $('qualityError').textContent='';
  await api('recording-note',{output,relative:r.relative,run_id:r.run_id,expected_revision:r.revision,note});
  $('qualityNote').value='';
  feedback.textContent='批注已保存至 '+(r.archive?.episode||r.relative)+'。';
 }catch(e){
  feedback.className='error';feedback.textContent='批注保存失败：'+e.message+'。输入内容已保留，请重试。';
 }
 finally{qualityBusy=false;await refreshRecordings(true);renderRecordingQuality()}
}
$('qualityNote').addEventListener('input',()=>{
 $('qualityNoteStatus').textContent='';renderRecordingQuality();
});
$('qualityDialog').addEventListener('cancel',event=>{event.preventDefault();if(!qualityBusy)closeQualityDialog()});

let agentPresets={},agentProviderCache={},lastAgentProvider='mengxiang';
const agentFields={provider:'agentProvider',repository:'agentRepository',checkpoint:'agentCheckpoint',python:'agentPython',pretrained:'agentPretrained',gello_repository:'agentGello',ros_overlay:'agentOverlay'};
function agentProviderChanged(){agentProviderCache[lastAgentProvider]={...agentConfig(),provider:lastAgentProvider};let provider=$('agentProvider').value,settings=agentProviderCache[provider]||agentPresets[provider];if(settings){for(let[k,id]of Object.entries(agentFields))$(id).value=settings[k]||'';}lastAgentProvider=provider;updateAgentChoice();$('agentError').textContent=''}
function agentConfig(){return {...Object.fromEntries(Object.entries(agentFields).map(([k,id])=>[k,$(id).value])),profile:'workstation'}}
function updateAgentChoice(){$('agentEnvironmentReport').textContent=''}
async function checkAgentEnvironment(){try{let r=await api('agent-check',agentConfig()),box=$('agentEnvironmentReport');box.replaceChildren();for(let c of r.checks){let line=document.createElement('div');line.textContent=(c.ok?'✓ ':'✗ ')+c.name+'：'+c.detail;line.className=c.ok?'muted':'error';box.append(line)}let note=document.createElement('p');note.textContent=r.note;box.append(note)}catch(e){$('agentError').textContent=e.message}}
let agentState=null,agentRequestPending=false;
function renderAgent(a){
if(!a)return;
agentState=a;

const canStart=!agentRequestPending&&['stopped','failed'].includes(a.status);
$('startAgent').disabled=!canStart;
$('stopAgent').disabled=['stopped','stopping'].includes(a.status);
for(const id of Object.values(agentFields))$(id).disabled=!canStart;
$('checkAgent').disabled=!canStart;
const messages={stopped:'Agent 未启动',loading:'正在加载',ready:'实时预测就绪',waiting:'等待实时观测',stopping:'正在停止',failed:'Agent 已退出'};
const detail=a.provider==='tairan'&&['waiting','ready'].includes(a.status)?a.message:messages[a.status];
$('agentStatus').textContent=[a.provider,detail,a.mode==='auto'?'自动切换':''].filter(Boolean).join(' · ');
if(a.status==='failed')$('agentError').textContent=a.message;
const box=$('agentProbabilities');
box.replaceChildren();
if(!['ready','waiting'].includes(a.status))return;
const horizon=a.future_horizon_s||(a.provider==='tairan'?1:1.5);
const show=text=>{const span=document.createElement('span');span.textContent=text;box.append(span)};
for(const [label,title] of [['current','当前预测'],['future','采集后 +'+horizon+' 秒']]){
 const values=a[label]?Object.entries(a[label]).map(([k,v])=>k+' '+(v*100).toFixed(1)+'%').join(' · '):'等待观测';
 show(title+'：'+values);
}
if(a.status==='ready'&&Number.isInteger(a.roi_fallback_frames))show('第四路 ROI 回退全图 '+a.roi_fallback_frames+'/16 帧');
if(a.provider==='tairan'&&a.status==='ready')show('推理耗时 '+Math.round(a.latency_ms)+' ms');
}
async function agentCommand(path){if(agentRequestPending)return;const settings=agentConfig();agentRequestPending=true;if(agentState)renderAgent(agentState);try{$('agentError').textContent='';renderAgent(await api(path,path==='start-agent'?settings:{}));await refresh()}catch(e){$('agentError').textContent=e.message}finally{agentRequestPending=false;if(agentState)renderAgent(agentState)}}
let taskState=null, taskPending=false, taskPolling=false, taskCatalogs={};
async function refreshTask(){if(!taskState||taskPolling||!['RUNNING','RESUMING','SWITCH_REQUESTED','SWITCHING','PAUSING','STOPPING'].includes(taskState.execution_state))return;taskPolling=true;try{renderTask(await api('task-status'))}catch(e){$('taskError').textContent=String(e)}finally{taskPolling=false}}
const oldStageList=[['pickup_1','首次拿起'],['harvest_1','首次采摘'],['return_truss','放回整串'],['adjust','调整'],['pickup_2','再次拿起'],['harvest_2','再次采摘'],['return_stem','放回裸杆']];
const inHandStageList=[['pickup','拿起番茄串'],['harvest_1','首次采摘'],['adjust_in_hand','手中调整'],['harvest_2','再次采摘'],['return_stem','最终放回']];
let stageList=oldStageList;
function configuredStageList(){
 return $('task_mode').value==='tomato-in-hand119'?inHandStageList:oldStageList;
}
function taskModeChanged(){
 updateDeploymentFields();

 const list=configuredStageList(), initial=$('initial_stage');
 if($('task_mode').value==='none'||!list.some(([id])=>id===initial.value)){
  initial.value=list[0][0];
  $('recovery_confirmed').checked=false;
 }
 updateAgentChoice();renderTask(taskState);
}
function selectInitialStage(id){
 if($('task_mode').value==='none'||$('initial_stage').value===id)return;
 $('initial_stage').value=id;
 $('recovery_confirmed').checked=false;
 invalidate();renderTask(taskState);
}
function renderTask(t){
 updateDeploymentFields();
 if(t&&taskState&&t.run_id===taskState.run_id&&t.stage_version<taskState.stage_version)return;
 taskState=t;
 const live=executionActive&&!executionStarting;
 const configuring=(!live||['STOPPING','STOPPED','COMPLETED','FAULT'].includes(t?.execution_state))&&$('task_mode').value!=='none';
 const configured=configuredStageList();
 const rows=(live?t?.stages:null)||taskCatalogs[$('task_mode').value]||taskCatalogs['tomato-b']||[];
 stageList=(live?t?.stages?.map(s=>[s.id,s.label]):null)||configured;
 const state=t?.execution_state||'IDLE', available=live&&!!t&&!t.communication_error;
 const initial=$('initial_stage').value;
 $('taskInstructions').textContent=configuring
  ?'点击下方阶段按钮选择初始阶段。'+(initial!==configured[0][0]?'从后续阶段启动需要勾选「人工恢复」并重新核对配置。':'')
  :state==='READY'?'选择阶段后，点击「开始当前阶段」。':'可点击任意阶段；Agent 启动后自动请求下一阶段，人工选择优先。';
 const box=$('taskStages');box.replaceChildren();
 stageList.forEach(([id,name])=>{
  const b=document.createElement('button');
  if(configuring){
   const selected=id===initial;
   b.textContent=name+(selected?' · 初始阶段':'');
   b.className=selected?'active':'';
   b.setAttribute('aria-pressed',String(selected));
   b.onclick=()=>selectInitialStage(id);
  }else{
   const active=live&&t?.active_stage===id, requested=live&&t?.requested_stage===id;
   const passed=live&&t?.completed_stages?.includes(id);
   b.textContent=(active?'● ':requested?'◷ ':'')+name+(passed?' · 已离开':'');
   b.className=active?'active':requested?'pending':passed?'passed':'';
   b.disabled=false;
   b.onclick=()=>selectLiveStage(id);
  }
  box.append(b);
 });
$('taskPrompt').textContent=t?.active_prompt||'尚未接入阶段动作';
const active=rows.find(stage=>stage.id===t?.active_stage);
$('autoActivePrompt').hidden=!t?.active_prompt;
$('autoPromptStage').textContent=(t?.communication_error?'上次确认的 Prompt':'当前激活的 Prompt')+(active?' · '+active.label:'');
$('autoPromptText').textContent=t?.active_prompt||'等待执行端接入首个动作块';
$('taskError').textContent=t?.communication_error||t?.last_error||'';
$('taskStart').textContent=state==='PAUSED'?'继续当前阶段':'开始当前阶段';
$('taskStart').disabled=!available||taskPending||!(state==='READY'||state==='PAUSED'&&t?.safe_to_resume);
$('taskPause').disabled=!available||taskPending||!['RUNNING','RESUMING','SWITCHING','SWITCH_REQUESTED'].includes(state);
if(daggerState&&executionActive){
 $('taskPause').disabled=true;
 if(t?.dagger_suspended||!['POLICY','IDLE'].includes(daggerState.mode)){
  $('taskStart').disabled=true;
 }
}}
let queuedManualStage=null,manualStageSending=false;
function selectLiveStage(id){
 queuedManualStage=id;
 return applyManualStage();
}
async function applyManualStage(){
 if(manualStageSending||taskPending||!queuedManualStage)return;
 manualStageSending=true;
 try{
  while(queuedManualStage&&!taskPending){
   const target=queuedManualStage;queuedManualStage=null;
   if(!executionActive||!taskState||['STOPPING','STOPPED','COMPLETED','FAULT'].includes(taskState.execution_state)){
    selectInitialStage(target);continue;
   }
   await taskCommand('stage',target);
  }
 }finally{manualStageSending=false}
}
async function taskCommand(command,target,confirmed=false){if(taskPending||!taskState)return;if(command==='resume'&&!confirm('确认安全状态并重新授权？恢复将获取新三路图像和关节状态，重新推理，绝不续播旧动作。'))return;const request={expected_run_id:taskState.run_id,client_clicked_unix_ms:Date.now(),command,request_id:crypto.randomUUID(),expected_version:taskState.stage_version,target,confirmed:confirmed||command==='resume'};taskPending=true;renderTask(taskState);try{taskState=await api('task-command',request);renderTask(taskState);$('error').textContent='';return true}catch(e){failure(e)}finally{taskPending=false;renderTask(taskState);await refreshTask();applyManualStage()}}

async function browse(target,file=false){if(busy)return;pickerTarget=target;pickerFile=file;$('chooseDirectory').hidden=file;$('browsePath').value=$(target).value||$('repository').value;$('picker').showModal();await loadDirectory()}
async function loadDirectory(){try{let d=await api('browse?path='+encodeURIComponent($('browsePath').value));$('browsePath').value=d.path;let box=$('entries');box.replaceChildren();for(let item of d.entries){let b=document.createElement('button');b.textContent=(item.directory?'📁 ':'📄 ')+item.name;b.onclick=()=>{if(item.directory){$('browsePath').value=item.path;loadDirectory()}else if(pickerFile)choosePath(item.path)};box.append(b)}}catch(e){failure(e)}}
function upDirectory(){let p=$('browsePath').value;$('browsePath').value=p.slice(0,p.lastIndexOf('/'))||'/';loadDirectory()}
function choosePath(path){$(pickerTarget).value=path;$('picker').close();invalidate();if(pickerTarget==='dagger_output')saveControlPreferences({dagger_output:path});if(pickerTarget==='repository')repositoryChanged();if(['checkpoint','training_record'].includes(pickerTarget)){modelTypeChanged();$('identity').textContent='权重配置已更改，请重新核对'}}
(async()=>{let d=await api('info');token=d.token;taskCatalogs=d.task_catalogs||{};agentPresets=d.agent_presets||{};lastAgentProvider=d.agent_defaults?.provider||'mengxiang';for(let [k,id]of Object.entries(agentFields))$(id).value=d.agent_defaults?.[k]||'';updateAgentChoice();for(let [k,v] of Object.entries(d.defaults||{}))if(fields.includes(k)){if(['record','recovery_confirmed','dagger'].includes(k))$(k).checked=v;else $(k).value=v}let savedSteps=$('steps').value;modeChanged();$('steps').value=savedSteps;for(let f of fields)$(f).addEventListener('input',invalidate);$('model_type').addEventListener('change',()=>modelTypeChanged(true));for(let f of ['checkpoint','training_record','train_rtc_max_delay','model_type','rtc_initial_delay_steps'])$(f).addEventListener('input',()=>{modelTypeChanged();$('identity').textContent='权重配置已更改，请重新核对';});$('repository').addEventListener('input',()=>{clearTimeout(repoTimer);repoTimer=setTimeout(repositoryChanged,350)});$('repository').addEventListener('change',repositoryChanged);$('image_input').addEventListener('change',()=>{inspectedViews=null;updateDeploymentFields();invalidate()});$('task_mode').addEventListener('change',taskModeChanged);taskModeChanged();modelTypeChanged();restoreControlPreferences(d.ui_preferences);await repositoryChanged();showLog('execution');await refresh();setInterval(refresh,1500);setInterval(refreshTask,250)})().catch(failure);
</script></html>"""


def server_for(root, *, port=8766):
    controller = DeploymentController(root)
    token = secrets.token_urlsafe(32)
    defaults = {"repository": str(root), "output": str(desktop() / "Sharpa部署记录"), "steps": 15, "rounds": 0, "task_mode": "none", "initial_stage": "pickup_1", "prompt": DEFAULT_PROMPT, "image_input": "auto",
                "model_type": "auto", "mode": "rtc", "speed": "fast", "initialization": "current"}
    with suppress(OSError, ValueError):
        defaults.update(json.loads((Path(root) / ".deployment/control_last.json").read_text()))

    # The UI uses the current DeploymentSettings defaults for these options.
    defaults.pop("camera_policy", None)
    defaults.pop("crop_device", None)
    defaults["recovery_confirmed"] = False  # Reauthorization is never restored from a saved checkbox.
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
                    self.send(200, {"defaults": defaults, "ui_preferences": dict(controller.ui_preferences), "token": token, "task_catalogs": {p: catalog(p) for p in ("tomato-b", "tomato-in-hand119")}, "agent_defaults": {**controller.agent.settings, "profile": "workstation"}, "agent_presets": {name: asdict(AgentSettings.parse({"provider": name, "profile": "workstation"}, root)) for name in AGENT_PROVIDERS}})
                elif parsed.path == "/api/task-status":
                    self.send(200, controller.task_status())
                elif parsed.path == "/api/crop-preview":
                    if not controller.plan or not controller.plan["model"].get("crop_locator"):
                        self.send(404, {"error": "No active crop model"})
                    else:
                        path = Path(controller.plan["settings"]["repository"])/".deployment/crop_live/preview.jpg"
                        self.send(200, path.read_bytes(), "image/jpeg")
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
                                                train_rtc_max_delay=data.get("train_rtc_max_delay", 0), model_type=data.get("model_type", "auto"), image_input=data.get("image_input", "auto"), task_mode=data.get("task_mode", "none"))
                elif self.path == "/api/preview":
                    result = controller.preview(data)
                elif self.path == "/api/preferences":
                    result = controller.save_ui_preferences(data)
                elif self.path == "/api/start-model":
                    result = controller.start_model(data["settings"])
                elif self.path == "/api/home-arms":
                    result = controller.home_arms(data.get("repository"))
                elif self.path == "/api/start":
                    if not data.get("approval"):
                        raise ValueError("请先核对最终配置")
                    result = controller.start(data["settings"], expected_approval=data["approval"])
                elif self.path == "/api/agent-check":
                    result = controller.agent.check_environment(data)
                elif self.path == "/api/start-agent":
                    result = controller.agent.start(data)
                elif self.path == "/api/stop-agent":
                    result = controller.agent.stop()
                elif self.path == "/api/agent-mode":
                    result = controller.agent.set_mode(data.get("mode"))
                elif self.path == "/api/task-command":
                    result = controller.task_command(data)
                elif self.path == '/api/dagger-command':
                    result = controller.dagger_command(data)
                elif self.path == "/api/recordings":
                    from .quality import list_recordings
                    result = list_recordings(data["output"])
                elif self.path == "/api/recording-note":
                    from .quality import update_note
                    result = update_note(data["output"], data["relative"], run_id=data["run_id"],
                                         note=data.get("note", ""), expected_revision=data["expected_revision"])
                elif self.path == "/api/rate-recording":
                    from .quality import rate
                    result = rate(data["output"], data["relative"], run_id=data["run_id"],
                                  grade=data["grade"], note=data.get("note", ""),
                                  expected_revision=data["expected_revision"])
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
            server.controller.agent.stop()
            if server.controller.agent.thread:
                server.controller.agent.thread.join(timeout=12)
            if server.controller.home_thread and server.controller.home_thread.is_alive():
                server.controller.stop()
            if server.controller.snapshot()["model_loading"]:
                server.controller.close_model()
            else:
                server.controller.stop()
            if server.controller.thread:
                server.controller.thread.join()  # Do not kill a device disable confirmation.
            if server.controller.home_thread:
                server.controller.home_thread.join()
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
