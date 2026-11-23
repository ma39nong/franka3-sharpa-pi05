"""Best-effort configuration hints; no training archive or registered identity required."""
import json
from pathlib import Path


def discover(checkpoint, record=""):
    root = Path(checkpoint)
    paths = [Path(record).expanduser()] if record else [
        root / "effective_config.json", root / "reproducibility/effective_config.json",
        root.parent / "reproducibility/effective_config.json", root / "train_rtc_record.json"]
    result = {"warnings": []}
    for path in paths:
        if not path.is_file():
            if record:
                result["warnings"].append(f"无法读取可选配置：{path}")
            continue
        try:
            raw = json.loads(path.read_text())
            fields = raw.get("config", raw).get("fields", {})
            model = fields.get("model", {}).get("fields", {})
            data = fields.get("data", {}).get("fields", {})
            if isinstance(data.get("extra_delta_transform"), bool):
                result["semantics"] = "delta" if data["extra_delta_transform"] else "absolute"
            delay = model.get("rtc_max_delay")
            if type(delay) is int and 0 <= delay < 50:
                result["rtc_max_delay"] = delay
            keys = model.get("image_keys")
            from .tomato_model import FOUR_IMAGE_KEYS
            if keys == list(FOUR_IMAGE_KEYS):
                result["image_input"] = "four"
            elif keys == list(FOUR_IMAGE_KEYS[:3]):
                result["image_input"] = "three"
            policy = fields.get("policy_metadata", {})
            prompts = list(policy.get("prompts", [])) + [p for d in raw.get("inputs", {}).get("datasets", []) for p in d.get("prompts", [])]
            if "Right hand adjusts tomato truss position and orientation at the center" in prompts:
                result["task_mode"] = "tomato-in-hand119"
            if policy.get("prompt_mode") == "global" and policy.get("global_prompt"):
                result["prompt"] = policy["global_prompt"]
                result["task_mode"] = "none"
            result["source"] = str(path)
            break
        except (ValueError, OSError, TypeError, AttributeError) as error:
            result["warnings"].append(f"可选配置无法识别，支持手动填写：{error}")
    return result


def input_route(views, task_mode="none", historical=None):
    if views != "four" and task_mode == "none":
        return None
    from .task_stages import TASK_CONTRACTS
    from .tomato_model import FOUR_IMAGE_KEYS
    result = dict(historical or {})
    # Historical labels are informational; routing follows the current settings.
    result.update(task_profile=task_mode, online_roi_required=views == "four",
                  image_keys=list(FOUR_IMAGE_KEYS if views == "four" else FOUR_IMAGE_KEYS[:3]),
                  task_name={"none": "统一 Prompt", "tomato-b": "放回盘中调整 · 七阶段", "tomato-in-hand119": "手中调整 · 五阶段"}.get(task_mode, task_mode),
                  input_name="四路 RGB（含局部图）" if views == "four" else "三路 RGB（全图）")
    if task_mode in TASK_CONTRACTS:
        result["stage_contract"] = TASK_CONTRACTS[task_mode]
    return result
