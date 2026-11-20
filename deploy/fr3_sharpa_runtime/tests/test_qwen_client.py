import json

import pytest

from deploy.fr3_sharpa_control import qwen_client


@pytest.mark.parametrize("fault", ["thinking", "empty", "incomplete", "timeout", "valid"])
def test_api_never_retries_or_accepts_thinking_and_incomplete_results(tmp_path, monkeypatch, fault):
    key = tmp_path / "key"
    key.write_text("fixture-secret")
    calls = []
    raw = {"choices": [{"finish_reason": "stop", "message": {"content": '{"events": []}'}}]}
    if fault == "thinking":
        raw["choices"][0]["message"]["reasoning_content"] = "unexpected thinking"
    elif fault == "empty":
        raw["choices"] = []
    elif fault == "incomplete":
        raw["choices"][0]["finish_reason"] = "length"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return json.dumps(raw).encode()

    class Opener:
        def open(self, request, **kwargs):
            payload = json.loads(request.data)
            calls.append(payload)
            if fault == "timeout":
                raise TimeoutError("fixture")
            return Response()

    monkeypatch.setattr(qwen_client.urllib.request, "build_opener", lambda *args: Opener())
    payload = {
        "model": "qwen3.8-flash",
        "enable_thinking": False,
        "messages": [{"role": "system", "content": "fixture prompt"}],
    }
    if fault == "valid":
        result = qwen_client.recognize(payload, key_path=key)
        assert result["answer"] == {"events": []}
        assert "fixture-secret" not in json.dumps(result)
    else:
        with pytest.raises(ValueError, match=".+"):
            qwen_client.recognize(payload, key_path=key)
    assert len(calls) == 1
    assert calls[0]["enable_thinking"] is False
