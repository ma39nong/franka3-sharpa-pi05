"""Bounded, non-thinking Qwen requests with explicit application memory."""

import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ENDPOINT = "https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1/chat/completions"
KEY_PATH = Path.home() / ".config/qwen-recognition-test/token-plan.key"
PROMPT_PATH = Path(__file__).with_name("qwen_online_prompt.txt")
MODELS = ("qwen3.8-flash", "qwen3.8-max", "qwen3.6-flash")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def build_payload(memory, request, images, *, model="qwen3.8-flash"):
    if model not in MODELS:
        raise ValueError("请选择已配置的非思考视觉模型:" + ", ".join(MODELS))
    if (
        request["run_id"] != memory["run_id"]
        or request["revision"] != memory["revision"]
        or request["active_stage"] != memory["active_stage"]
    ):
        raise ValueError("请求与任务记忆不一致")
    frames = request["frames"]
    if not 2 <= len(frames) <= 64 or len({f["id"] for f in frames}) != len(frames):
        raise ValueError("每次请求须提供2至64张有唯一编号的画面")
    context = {"task_memory": memory, "observation": request}
    content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False, allow_nan=False)}]
    total = 0
    for frame in frames:
        blob = images[frame["id"]]
        total += len(blob)
        if not blob.startswith(b"\xff\xd8") or total > 12 * 1024 * 1024:
            raise ValueError("输入须为JPEG,总大小不超过12MiB")
        content.extend(
            (
                {"type": "text", "text": json.dumps(frame, ensure_ascii=False, allow_nan=False)},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(blob).decode()},
                },
            )
        )
    return {
        "model": model,
        "enable_thinking": False,
        "stream": False,
        "max_tokens": 1800,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": PROMPT_PATH.read_text()}, {"role": "user", "content": content}],
    }


def recognize(payload, *, key_path=KEY_PATH, timeout=20.0):
    if payload.get("enable_thinking") is not False or payload.get("model") not in MODELS:
        raise ValueError("只允许显式关闭思考的已配置视觉模型")
    key = Path(key_path).read_text().strip()
    if not key:
        raise ValueError("Qwen密钥未配置")
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=encoded,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
    )
    start = time.monotonic()
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as response:
            blob = response.read(2 * 1024 * 1024 + 1)
            if len(blob) > 2 * 1024 * 1024:
                raise ValueError("Qwen响应超出限制")
            raw = json.loads(blob)
    except urllib.error.HTTPError as error:
        raise ValueError(f"Qwen HTTP {error.code};未重试、未切换端点或思考模式") from None
    except (OSError, TimeoutError) as error:
        raise ValueError("Qwen请求超时或连接失败,未重试") from error
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Qwen缺少有效回答")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Qwen输出不完整,未采用")
    reasoning = raw.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    if choice.get("message", {}).get("reasoning_content") or reasoning:
        raise ValueError("服务端返回了思考内容,不符合关闭思考的请求要求")
    answer = json.loads(choice["message"]["content"])
    if not isinstance(answer, dict):
        raise ValueError("Qwen未返回JSON对象")
    return {
        "answer": answer,
        "requested_model": payload["model"],
        "reported_model": raw.get("model"),
        "elapsed_s": time.monotonic() - start,
        "usage": raw.get("usage"),
        "prompt_sha256": hashlib.sha256(payload["messages"][0]["content"].encode()).hexdigest(),
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "raw": raw,
    }
