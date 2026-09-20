"""Serve 25000 with its own state transform and the arm-first 64D action map."""

import argparse
import dataclasses
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from deploy.fr3_wuji_models.model_25000.contract import DEFAULT_CHECKPOINT
from deploy.fr3_wuji_models.model_25000.contract import checkpoint_contract
from deploy.fr3_wuji_models.model_30000 import serve as base


@dataclasses.dataclass(frozen=True)
class PhysicalOrderedState:
    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (54,) or not np.isfinite(state).all():
            raise ValueError("Expected one finite 54D physical state")
        return {**data, "state": np.pad(state, (0, 10))}


@dataclasses.dataclass(frozen=True)
class AdaptData:
    base_data: object

    def create(self, assets_dirs, model_config):
        from openpi import transforms

        data = dataclasses.replace(self.base_data, extra_delta_transform=False).create(assets_dirs, model_config)
        group = transforms.Group(
            inputs=(*data.data_transforms.inputs, PhysicalOrderedState()),
            outputs=(base.PhysicalAbsoluteActions(),),
        )
        return dataclasses.replace(data, data_transforms=group)


def create_policy(checkpoint, stats_file, prompt, *, contract=checkpoint_contract):
    from openpi.policies import policy_config
    from openpi.training import checkpoints
    from openpi.training import config

    original = config.get_config("pi05_fr3_wuji")
    model = base.full_finetune_model(original.model)
    adapted = dataclasses.replace(
        original,
        name="pi05_fr3_wuji_25000_full_64to54",
        model=model,
        data=AdaptData(original.data),
    )
    stats = checkpoints.load_norm_stats(
        checkpoint / "assets", str(stats_file.parent.relative_to(checkpoint / "assets"))
    )
    policy = policy_config.create_trained_policy(adapted, checkpoint, norm_stats=stats, default_prompt=prompt)
    metadata, _ = contract(checkpoint)
    return base.CheckedPolicy(policy, metadata)


def main(argv=None, *, contract=checkpoint_contract, default_checkpoint=DEFAULT_CHECKPOINT, default_port=8004):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--prompt", default=base.PROMPT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    metadata, stats_file = contract(args.checkpoint)
    checkpoint = Path(metadata["checkpoint"])
    print(f"Checkpoint contract OK: {stats_file}", flush=True)
    if args.check_only:
        return
    import jax

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("CUDA GPU required")
    policy = create_policy(checkpoint, stats_file, args.prompt, contract=contract)
    observation = {"observation/state": np.zeros(54, np.float32), "prompt": args.prompt}
    for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
        observation[key] = np.zeros((224, 224, 3), np.uint8)
    result = policy.infer(observation)
    print(f"Warmup OK: actions={result['actions'].shape}", flush=True)
    policy.metadata["warmed_up"] = True
    if args.smoke_only:
        return
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    print(f"Serving {metadata['config']} at ws://{args.host}:{args.port}", flush=True)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
