"""CPU-only isolation and checkpoint-contract tests."""

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deploy.fr3_wuji_30000 import serve as lora_serve
from deploy.fr3_wuji_30000v2 import execute
from deploy.fr3_wuji_30000v2 import serve
from deploy.fr3_wuji_30000v2.contract import checkpoint_contract


def test_checkpoint_is_full_finetune_and_64d():
    checkpoint = ROOT / "checkpoints/30000v2"
    tree = json.loads((checkpoint / "params/_METADATA").read_text())["tree_metadata"]
    assert not any("lora" in key.lower() for key in tree)
    metadata, stats_file = checkpoint_contract(checkpoint)
    assert metadata["checkpoint"] == str(checkpoint.resolve())
    assert metadata["model_finetune_mode"] == "full"
    assert metadata["model_action_dim"] == 64
    assert metadata["action_dim"] == 54
    assert metadata["action_representation"] == "absolute_joint_positions"
    stats = json.loads(stats_file.read_text())["norm_stats"]
    assert all(len(stats[key][field]) == 64 for key in ("state", "actions") for field in stats[key])
    assert all(all(value == 0 for value in stats[key][field][54:])
               for key in ("state", "actions") for field in stats[key])


def test_server_defaults_are_isolated():
    assert serve.with_defaults(["--check-only"]) == [
        "--check-only", "--checkpoint", str(ROOT / "checkpoints/30000v2"),
        "--port", "8003", "--full-finetune",
    ]
    assert serve.with_defaults(["--port=9000", "--checkpoint=/tmp/x"]) == [
        "--port=9000", "--checkpoint=/tmp/x", "--full-finetune",
    ]


def test_slow_executor_defaults_are_isolated():
    assert execute.with_defaults(["--check"]) == [
        "--check", "--checkpoint", str(ROOT / "checkpoints/30000v2"),
        "--uri", "ws://127.0.0.1:8003",
    ]
    old, _ = lora_serve.checkpoint_contract(ROOT / "checkpoints/30000")
    new, _ = checkpoint_contract(ROOT / "checkpoints/30000v2")
    assert old["checkpoint"] != new["checkpoint"]
    assert old["checkpoint_manifest_sha256"] != new["checkpoint_manifest_sha256"]
    assert old["config"] != new["config"]
