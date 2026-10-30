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
*{box-sizing:border-box}body{margin:0;background:#eff3f6;color:#15293e;font:15px/1.6 system-ui,sans-serif}
header{background:#132f40;color:white;padding:22px 5%;display:flex;justify-content:space-between;align-items:center}
h1{font-size:24px;margin:0}header p{margin:4px 0;color:#bbd1da}main{max-width:1280px;margin:24px auto;padding:0 24px}
.grid{display:grid;grid-template-columns:1.05fr 1fr;gap:20px}.card{background:white;border:1px solid #dae3e9;border-radius:12px;padding:22px;margin-bottom:18px}
h2{font-size:18px;margin:0 0 16px}label{display:block;margin-top:12px;font-weight:600}input,select{width:100%;padding:10px;border:1px solid #b9c8d2;border-radius:6px;background:white;font:inherit;color:inherit}
input[type=checkbox]{width:auto}.row{display:flex;gap:10px;align-items:end}.row>*{flex:1}.path{display:flex;gap:6px}.path input{min-width:0}.path button{flex:0 0 auto}
button{border:0;border-radius:6px;padding:10px 14px;background:#dfe9ee;color:#17394a;font:600 14px system-ui;cursor:pointer;white-space:nowrap}button.primary{background:#087b75;color:white}button.stop{background:#b63f37;color:white}button:disabled{opacity:.45;cursor:not-allowed}
.muted{color:#62778a;font-size:13px}.notice{background:#e9f5f3;padding:12px;border-radius:6px;margin:12px 0}.error{color:#b7352f;white-space:pre-wrap}.badge{padding:6px 12px;background:#254c60;border-radius:20px;font-size:13px}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,monospace;margin:8px 0;max-height:330px;overflow:auto}.logs pre{background:#132b38;color:#d5e5ed;padding:14px;border-radius:6px}
.actions{display:flex;gap:10px;flex-wrap:wrap}dialog{border:0;border-radius:10px;width:760px;max-width:95vw}#entries{height:350px;overflow:auto}#entries button{display:block;background:white;text-align:left;width:100%;border-bottom:1px solid #e5edf1}details{margin:12px 0}summary{cursor:pointer;font-weight:600}.preview{min-height:100px}
@media(max-width:850px){.grid{grid-template-columns:1fr}header{align-items:start;gap:10px}}
</style><header><div><h1>Sharpa 部署控制台</h1><p>权重身份 · 执行方式 · 录制归档</p></div><span id="phase" class="badge">空闲</span></header>
<main><div class="grid"><section><div class="card"><h2>1 · 权重与代码</h2>
<label>部署仓库目录</label><div class="path"><input id="repository"><button onclick="browse('repository')">选择目录</button></div>
<div class="muted" id="repoInfo">Main 与 Soft 可各用一个部署目录；不会在运行中切换 Git 分支。</div>
<label>Checkpoint 步骤目录</label><div class="path"><input id="checkpoint" placeholder="直接包含 params/ 和 assets/ 的目录"><button onclick="browse('checkpoint')">选择目录</button></div>
<details><summary>训练证据 / 无法自动识别时</summary><label>归档训练 JSON（可选）</label><div class="path"><input id="training_record"><button onclick="browse('training_record',true)">选择文件</button></div>
<label>动作语义</label><select id="semantics"><option value="auto">自动核对，不按目录名猜测</option><option value="absolute">我按训练配置确认是 Absolute</option><option value="delta">我按训练配置确认：双臂 Delta，双手 Absolute</option></select></details>
<button onclick="inspect()" style="margin-top:14px">识别权重</button><div id="identity" class="notice">覆盖：30Hz／58D 的全量 Absolute、普通 Delta、Paper Delta，以及 Absolute LoRA。Paper按资产识别；LoRA Delta与其他结构尚不支持。实际结构在GPU加载时再次核对。</div>
</div><div class="card"><h2>2 · 本次执行配置</h2>
<div class="row"><div><label>执行模式</label><select id="mode" onchange="modeChanged()"><option value="serial">普通串行</option><option value="rtc">RTC</option><option value="soft-serial">串行 Soft</option><option value="soft-async">异步 Soft · 50→30→26＋4→20</option></select></div><div><label>启动姿态</label><select id="initialization"><option value="current">Current · 从当前状态推理</option><option value="home">Home · 先回固定位置再推理</option></select></div></div>
<div class="row"><div><label>速度</label><select id="speed"><option value="slow">Slow · 0.1倍</option><option value="medium">Medium · 0.5倍</option><option value="fast">Fast · 模型原速</option></select></div><div><label id="stepsLabel">每轮执行的模型动作数</label><input id="steps" type="number" value="15" min="1" max="50"></div><div><label>轮数（0为持续）</label><input id="rounds" type="number" value="0" min="0"></div></div>
<div id="modeHint" class="notice"></div><label><input id="record" type="checkbox"> 录制 ROSbag，并在结束后生成三路及三联视频</label>
<label>归档父目录</label><div class="path"><input id="output"><button onclick="browse('output')">选择目录</button></div><div class="muted">自动创建 日期/01、02…；不录制时只保留配置和运行日志。不会覆盖旧实验。</div>
</div></section><section><div class="card"><h2>3 · 核对并启动</h2><div class="actions"><button id="preview" onclick="preview()">核对最终配置</button><button id="start" class="primary" onclick="start()" disabled>启动部署</button><button id="stop" class="stop" onclick="stop()" disabled>停止执行</button><button id="closeModel" onclick="closeModel()">关闭模型服务</button></div>
<p class="muted">启动自动完成 up → 模型预热与身份核对 → 执行。停止等效一次 Ctrl-C，等待失能确认与清理后才能再次启动。停止不会自动回 Home。</p><div id="message" class="notice">尚未启动设备</div><div id="error" class="error"></div><pre id="previewText" class="preview">请选择配置后点击“核对最终配置”。</pre><div id="directory" class="muted"></div><pre id="recordStatus"></pre></div>
<div class="card logs"><h2>运行日志</h2><div id="logs"></div></div><div class="card muted">三路实时预览留待下一阶段。本版先统一配置、启停、录制和离线视频；打开页面不会连接设备。关闭页面也不等于停止执行，请使用停止按钮。</div></section></div></main>
<dialog id="picker"><h2>选择工作机上的路径</h2><div class="path"><input id="browsePath"><button onclick="loadDirectory()">打开</button><button onclick="upDirectory()">上一级</button></div><div id="entries"></div><div class="actions"><button class="primary" id="chooseDirectory" onclick="choosePath(document.getElementById('browsePath').value)">选择此目录</button><button onclick="document.getElementById('picker').close()">取消</button></div></dialog>
<script>
let token='', approved=null, approvedPlan=null, busy=false, pickerTarget='', pickerFile=false, repoModes=['serial','rtc'], modelModes=null, previousMode='serial', repoTimer=null;
const $=id=>document.getElementById(id);
const fields=['repository','checkpoint','mode','initialization','speed','steps','rounds','record','output','training_record','semantics'];
function config(){let c={};for(let f of fields)c[f]=f==='record'?$(f).checked:['steps','rounds'].includes(f)?Number($(f).value):$(f).value;return c}
async function api(path,data){const r=await fetch('/api/'+path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':token},body:JSON.stringify(data)});let d=await r.json();if(!r.ok)throw Error(d.error);return d}
function failure(e){$('error').textContent=e.message}
function invalidate(){approved=null;approvedPlan=null;$('start').disabled=true;$('error').textContent=''}
function modeChanged(){let mode=$('mode').value,soft=mode.startsWith('soft-');$('steps').disabled=soft;$('steps').value=soft?26:15;$('steps').max=mode==='rtc'?24:50;$('stepsLabel').textContent=soft?'源执行点（固定26）':mode==='rtc'?'重规划触发源节点':'每轮执行的模型动作数';$('speed').disabled=mode==='rtc';if(mode==='rtc')$('speed').value='fast';else if(previousMode==='rtc')$('speed').value='slow';previousMode=mode;$('modeHint').textContent=mode==='soft-async'?'固定 50→30→26＋4→20；20是压缩后的目标数，不是重规划15点。':mode==='soft-serial'?'50→30→26＋4；26压成20，原串行压缩策略可能回退26。':mode==='rtc'?'原速后台重规划；此处节点数不是串行的固定执行前缀。':'每轮预测50，执行填写的前N个动作，然后重新观测推理。';invalidate()}
function modeNames(modes){return modes.map(m=>Array.from($('mode').options).find(o=>o.value===m).textContent).join('、')}
function updateModes(){let modes=repoModes.filter(m=>!modelModes||modelModes.includes(m));for(let o of $('mode').options)o.disabled=!modes.includes(o.value);if(!modes.includes($('mode').value)){$('mode').value=modes[0]||'serial';modeChanged()}if(modelModes)$('identity').textContent=$('identity').textContent.replace(/可用模式：.*/, '可用模式：'+modeNames(modes));return modes}
async function inspect(){try{const c=config(),d=await api('inspect',c);modelModes=d.supported_modes;const modes=updateModes();$('identity').textContent=`${d.finetune_mode} · ${d.semantics||'语义未知'} · 30 Hz / 58D / horizon 50\n${d.evidence}\n归一化：${d.normalization}；噪声：${d.paper_algorithm?'相关噪声 β='+d.paper_algorithm.beta:'独立高斯'}\n可用模式：${modeNames(modes)}`;invalidate()}catch(e){failure(e)}}
async function repositoryChanged(){try{let d=await api('repository?path='+encodeURIComponent($('repository').value));repoModes=d.modes;$('repoInfo').textContent='当前分支：'+d.branch+'；仅可选当前代码支持的模式';updateModes();invalidate()}catch(e){failure(e)}}
async function preview(){try{const c=config(),p=await api('preview',c);$('previewText').textContent=JSON.stringify({分支:p.branch,提交:p.commit.slice(0,8),权重:p.model.checkpoint,模型:p.model.profile,证据:p.model.evidence,归一化:p.model.normalization,论文噪声:p.model.paper_algorithm,方式:p.description,姿态:c.initialization,速度:c.speed,轮数:c.rounds,录制:c.record,归档:p.settings.output},null,2);approved=JSON.stringify(c);approvedPlan=p.approval;$('start').disabled=busy;$('error').textContent=''}catch(e){invalidate();failure(e)}}
async function start(){try{if(approved!==JSON.stringify(config()))throw Error('配置已改变，请重新核对');await api('start',{settings:config(),approval:approvedPlan});busy=true;$('start').disabled=true;await refresh()}catch(e){failure(e)}}
async function stop(){try{await api('stop',{});await refresh()}catch(e){failure(e)}}
async function closeModel(){try{await api('close-model',{});await refresh()}catch(e){failure(e)}}
async function refresh(){try{let s=await api('status');busy=s.running;$('phase').textContent=({idle:'空闲',validating:'配置检查',hardware_up:'启动硬件',model_loading:'模型加载',recording_ready:'准备录制',running:'执行中',stopping:'停止中',exporting:'视频导出',failed:'失败',stopped:'已停止',finished:'已完成'})[s.phase]||s.phase;$('message').textContent=s.message;$('stop').disabled=!busy||s.stop_requested;$('start').disabled=busy||approved!==JSON.stringify(config());$('closeModel').disabled=busy||!s.model_running;$('directory').textContent=s.directory?'归档目录：'+s.directory:'';$('recordStatus').textContent=JSON.stringify({录制:s['recording.json'],本次视频:s['video_export.json'],后台导出:s.exports},null,2);let box=$('logs'),opened={};for(let d of box.querySelectorAll('details')){let p=d.querySelector('pre');opened[d.dataset.log]={open:d.open,top:p.scrollTop,end:p.scrollTop+p.clientHeight>=p.scrollHeight-8}}box.replaceChildren();for(let [name,text] of Object.entries(s.logs||{})){let details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent=name;pre.textContent=text;details.append(summary,pre);details.dataset.log=name;details.open=opened[name]===undefined?['runtime.log','model.log'].includes(name):opened[name].open;box.append(details);pre.scrollTop=opened[name]&&!opened[name].end?opened[name].top:pre.scrollHeight}for(let f of fields)$(f).disabled=busy; if(!busy){$('steps').disabled=$('mode').value.startsWith('soft-');$('speed').disabled=$('mode').value==='rtc'}}catch(e){failure(e)}}
async function browse(target,file=false){if(busy)return;pickerTarget=target;pickerFile=file;$('chooseDirectory').hidden=file;$('browsePath').value=$(target).value||$('repository').value;$('picker').showModal();await loadDirectory()}
async function loadDirectory(){try{let d=await api('browse?path='+encodeURIComponent($('browsePath').value));$('browsePath').value=d.path;let box=$('entries');box.replaceChildren();for(let item of d.entries){let b=document.createElement('button');b.textContent=(item.directory?'📁 ':'📄 ')+item.name;b.onclick=()=>{if(item.directory){$('browsePath').value=item.path;loadDirectory()}else if(pickerFile)choosePath(item.path)};box.append(b)}}catch(e){failure(e)}}
function upDirectory(){let p=$('browsePath').value;$('browsePath').value=p.slice(0,p.lastIndexOf('/'))||'/';loadDirectory()}
function choosePath(path){$(pickerTarget).value=path;$('picker').close();invalidate();if(pickerTarget==='repository')repositoryChanged();if(['checkpoint','training_record'].includes(pickerTarget)){modelModes=null;updateModes();$('identity').textContent='权重配置已更改，请重新识别'}}
(async()=>{let d=await api('info');token=d.token;for(let [k,v] of Object.entries(d.defaults||{}))if(fields.includes(k)){if(k==='record')$(k).checked=v;else $(k).value=v}let savedSteps=$('steps').value;modeChanged();if(!$('mode').value.startsWith('soft-'))$('steps').value=savedSteps;for(let f of fields)$(f).addEventListener('input',invalidate);for(let f of ['checkpoint','training_record','semantics'])$(f).addEventListener('input',()=>{modelModes=null;updateModes();$('identity').textContent='权重配置已更改，请重新识别';});$('repository').addEventListener('input',()=>{clearTimeout(repoTimer);repoTimer=setTimeout(repositoryChanged,350)});$('repository').addEventListener('change',repositoryChanged);await repositoryChanged();await refresh();setInterval(refresh,1500)})().catch(failure);
</script></html>"""


def server_for(root, *, port=8766):
    controller = DeploymentController(root)
    token = secrets.token_urlsafe(32)
    defaults = {"repository": str(root), "output": str(desktop() / "Sharpa部署记录"), "steps": 15, "rounds": 0}
    with suppress(OSError, ValueError):
        defaults.update(json.loads((Path(root) / ".deployment/control_last.json").read_text()))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def send(self, status, data, kind="application/json; charset=utf-8"):
            body = data.encode() if isinstance(data, str) else json.dumps(data, ensure_ascii=False).encode()
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
                elif parsed.path == "/api/info":
                    self.send(200, {"defaults": defaults, "token": token})
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
                                                semantics=data.get("semantics", "auto"), cache=controller.cache)
                elif self.path == "/api/preview":
                    result = controller.preview(data)
                elif self.path == "/api/start":
                    if not data.get("approval"):
                        raise ValueError("请先核对最终配置")
                    result = controller.start(data["settings"], expected_approval=data["approval"])
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
            server.controller.stop()
            if server.controller.thread:
                server.controller.thread.join()  # Do not kill a device disable confirmation.
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
