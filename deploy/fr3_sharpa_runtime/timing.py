"""Read-only phase timing for the local Sharpa server; outputs stay unchanged."""
import time


class TimedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.received_ns = None

    def infer(self, obs, **kwargs):
        timing = {"server_receive_monotonic_ns": self.received_ns,
                  "policy_inference_start_monotonic_ns": time.monotonic_ns()}
        original = {}
        for attr, label in (("_input_transform", "preprocessing"),
                            ("_sample_actions", "model_sampling"),
                            ("_output_transform", "postprocessing")):
            function = getattr(self.policy, attr)
            original[attr] = function
            def timed(*args, _fn=function, _label=label, **kwargs):
                start = time.monotonic_ns()
                result = _fn(*args, **kwargs)
                if _label == "model_sampling":
                    # Await async device work only for measurement; preserve values.
                    import jax
                    jax.block_until_ready(result)
                end = time.monotonic_ns()
                timing[_label + "_start_monotonic_ns"] = start
                timing[_label + "_end_monotonic_ns"] = end
                timing[_label + "_ms"] = (end-start)/1e6
                return result
            setattr(self.policy, attr, timed)
        try:
            result = self.policy.infer(obs, **kwargs)
        finally:
            for attr, function in original.items():
                setattr(self.policy, attr, function)
        timing["policy_inference_end_monotonic_ns"] = time.monotonic_ns()
        timing["policy_total_ms"] = (timing["policy_inference_end_monotonic_ns"]-timing["policy_inference_start_monotonic_ns"])/1e6
        result["diagnostic_timing"] = timing
        return result


def timed_server_class(base):
    class TimedServer(base):
        async def _handler(self, websocket):
            policy = self._policy
            class Connection:
                def __getattr__(self, name):
                    return getattr(websocket, name)
                async def recv(self):
                    result = await websocket.recv()
                    policy.received_ns = time.monotonic_ns()
                    return result
            return await super()._handler(Connection())
    return TimedServer
