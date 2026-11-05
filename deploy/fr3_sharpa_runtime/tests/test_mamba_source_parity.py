"""Optional same-weight comparisons against the pinned, separate training checkout.

Set SHARPA_MAMBA_SOURCE to run. Synthetic full H50/D58 fixtures are explicitly
separate from real trained-checkpoint acceptance.
"""

import os
from pathlib import Path
import subprocess
import sys

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.mamba_contract import SOURCE_COMMIT
from deploy.fr3_sharpa_runtime.mamba_model import MambaPolicy
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_config
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_network
from openpi import transforms
from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from openpi.shared.normalize import NormStats

REFERENCE = r"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from scripts.mamba3_diagnostics import diagnostic_config, diagnostic_network
from openpi.models import model as model_lib
from openpi.policies.mamba_policy import MambaOfflinePolicy
from openpi.shared.normalize import NormStats
from openpi import transforms

with diagnostic_network():
    cfg = replace(diagnostic_config(), rtc_max_delay=3)
    model = cfg.create(jax.random.key(42))
    for block in model.action_expert.blocks.values():
        block.film_out.kernel.value = jax.random.normal(jax.random.key(31), block.film_out.kernel.shape) * .03
        block.cond_proj.kernel.value = jnp.eye(16) * .02
    obs = cfg.fake_obs()
    obs = replace(obs, images={k:jax.random.uniform(jax.random.key(i+3),v.shape,minval=-1,maxval=1) for i,(k,v) in enumerate(obs.images.items())})
    noise = jax.random.normal(jax.random.key(6), (1,50,58))
    prefix = jax.random.normal(jax.random.key(7), (1,50,58))
    values = {'noise':np.asarray(noise), 'prefix':np.asarray(prefix)}
    values.update({'p/'+ '/'.join(k):np.asarray(v) for k,v in traverse_util.flatten_dict(nnx.state(model).to_pure_dict()).items() if v is not None})
    values.update({'o/'+ '/'.join(k):np.asarray(v) for k,v in traverse_util.flatten_dict(obs.to_dict()).items() if v is not None})
    values['hidden'] = np.asarray(model.prefix_hidden(obs)[0])
    condition = model.encode_condition(obs)
    values['compressor'] = np.asarray(model.context_compressor(*model.prefix_hidden(obs)))
    for d in (0,1,3):
        values['sample_'+str(d)] = np.asarray(model.sample_actions(jax.random.key(2),obs,noise=noise,num_steps=2,prefix_actions=prefix,prefix_length=d))
    stat = NormStats(mean=np.linspace(-.1,.1,58),std=np.linspace(.2,2.,58),q01=-np.ones(58),q99=np.ones(58))
    data = SimpleNamespace(norm_stats={'state':stat,'actions':stat},use_quantile_norm=False,
        data_transforms=transforms.Group(),model_transforms=transforms.Group(),repack_transforms=transforms.Group())
    policy = MambaOfflinePolicy(model,data)
    raw = jax.tree.map(lambda x:np.asarray(x)[0],obs.to_dict())
    values['physical'] = policy.infer(raw,rng=jax.random.key(2),noise=noise,num_steps=2)['actions'][0]
    np.savez(sys.argv[1],**values)
"""


def test_pinned_training_same_weights_full_forward_and_physical_policy(tmp_path):
    source = os.environ.get("SHARPA_MAMBA_SOURCE")
    if not source:
        pytest.skip("Set SHARPA_MAMBA_SOURCE for the pinned training/deployment comparison")
    source = Path(source).resolve()
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == SOURCE_COMMIT
    output = tmp_path / "training_reference.npz"
    env = {**os.environ, "PYTHONPATH": f"{source}/src:{source}/packages/openpi-client/src:{source}"}
    result = subprocess.run(
        [sys.executable, "-c", REFERENCE, str(output)], cwd=source, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    with tiny_network(), np.load(output) as expected:
        cfg = tiny_config(rtc_max_delay=3)
        model = cfg.create(jax.random.key(0))
        graph, state = nnx.split(model)
        flat = traverse_util.flatten_dict(state.to_pure_dict())
        flat = {k: jnp.asarray(expected["p/" + "/".join(k)]) if v is not None else None for k, v in flat.items()}
        state.replace_by_pure_dict(traverse_util.unflatten_dict(flat))
        model = nnx.merge(graph, state)
        obs_dict = traverse_util.unflatten_dict(
            {tuple(k[2:].split("/")): expected[k] for k in expected.files if k.startswith("o/")}
        )
        obs = model_lib.Observation.from_dict(jax.tree.map(jnp.asarray, obs_dict))
        np.testing.assert_allclose(model.prefix_hidden(obs)[0], expected["hidden"], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            model.context_compressor(*model.prefix_hidden(obs)), expected["compressor"], rtol=1e-5, atol=1e-5
        )
        sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("prefix_length",))
        for delay in (0, 1, 3):
            actual = sample(
                jax.random.key(2),
                obs,
                noise=expected["noise"],
                num_steps=2,
                prefix_actions=expected["prefix"],
                prefix_length=delay,
            )
            np.testing.assert_allclose(actual, expected[f"sample_{delay}"], rtol=1e-4, atol=1e-4)
        stat = NormStats(
            mean=np.linspace(-0.1, 0.1, 58), std=np.linspace(0.2, 2.0, 58), q01=-np.ones(58), q99=np.ones(58)
        )
        stats = {"state": stat, "actions": stat}
        policy = MambaPolicy(
            model,
            transforms=[transforms.Normalize(stats)],
            output_transforms=[transforms.Unnormalize({"actions": stat})],
            sample_kwargs={"num_steps": 2},
        )
        raw = jax.tree.map(lambda x: np.asarray(x)[0], obs.to_dict())
        actual = policy.infer(raw, noise=expected["noise"])["actions"]
        np.testing.assert_allclose(actual, expected["physical"], rtol=1e-4, atol=1e-4)
