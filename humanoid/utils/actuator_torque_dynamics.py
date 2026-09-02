"""Discrete actuator-torque channel models shared by simulation paths.

The models map ideal joint-side PD torque to the torque applied by the
simulator.  They are not mechanical joint models and therefore do not replace
inertia, friction, damping, contact, or transmission coupling.
"""

import math

import numpy as np


MODEL_NONE = 0
MODEL_STATIC_GAIN = 1
MODEL_FOPDT = 2

_MODEL_CODES = {
    "none": MODEL_NONE,
    "static_gain": MODEL_STATIC_GAIN,
    "fopdt": MODEL_FOPDT,
}


def resolve_actuator_torque_dynamics(dof_names, specs, dt):
    """Resolve per-joint model specifications into simulation-step arrays."""
    if dt <= 0.0:
        raise ValueError("actuator torque dynamics dt must be positive")

    names = list(dof_names)
    specs = specs or {}
    unknown = sorted(set(specs) - set(names))
    if unknown:
        raise ValueError(
            "actuator torque dynamics contains unknown joints: " + ", ".join(unknown)
        )

    count = len(names)
    model_codes = np.zeros(count, dtype=np.int64)
    delay_s = np.zeros(count, dtype=np.float64)
    time_constant_s = np.zeros(count, dtype=np.float64)
    gain = np.ones(count, dtype=np.float64)

    for index, name in enumerate(names):
        if name not in specs:
            continue
        spec = specs[name]
        model_name = str(spec.get("model", "none")).lower()
        if model_name not in _MODEL_CODES:
            raise ValueError(
                "unsupported actuator torque model {!r} for {}".format(model_name, name)
            )

        model_codes[index] = _MODEL_CODES[model_name]
        gain[index] = float(spec.get("gain", 1.0))
        delay_s[index] = float(spec.get("delay_ms", 0.0)) * 1e-3
        time_constant_s[index] = float(spec.get("time_constant_ms", 0.0)) * 1e-3

        if gain[index] < 0.0:
            raise ValueError("actuator torque gain must be non-negative for " + name)
        if delay_s[index] < 0.0:
            raise ValueError("actuator torque delay must be non-negative for " + name)
        if model_codes[index] == MODEL_FOPDT and time_constant_s[index] <= 0.0:
            raise ValueError("FOPDT time constant must be positive for " + name)

    delay_steps = delay_s / float(dt)
    delay_floor = np.floor(delay_steps).astype(np.int64)
    delay_fraction = delay_steps - delay_floor
    max_delay_steps = int(math.ceil(float(delay_steps.max()))) if count else 0
    alpha = np.ones(count, dtype=np.float64)
    fopdt = model_codes == MODEL_FOPDT
    alpha[fopdt] = 1.0 - np.exp(-float(dt) / time_constant_s[fopdt])

    return {
        "model_codes": model_codes,
        "gain": gain,
        "delay_floor": delay_floor,
        "delay_fraction": delay_fraction,
        "alpha": alpha,
        "max_delay_steps": max_delay_steps,
    }


class NumpyActuatorTorqueDynamics:
    """Stateful NumPy implementation used by MuJoCo sim2sim inference."""

    def __init__(self, dof_names, specs, dt):
        params = resolve_actuator_torque_dynamics(dof_names, specs, dt)
        self.model_codes = params["model_codes"]
        self.gain = params["gain"]
        self.delay_floor = params["delay_floor"]
        self.delay_fraction = params["delay_fraction"]
        self.alpha = params["alpha"]
        self.buffer = np.zeros(
            (len(self.model_codes), params["max_delay_steps"] + 2), dtype=np.float64
        )
        self.state = np.zeros(len(self.model_codes), dtype=np.float64)

    @property
    def enabled(self):
        return bool(np.any(self.model_codes != MODEL_NONE))

    def reset(self):
        self.buffer.fill(0.0)
        self.state.fill(0.0)

    def update(self, ideal_torque):
        ideal_torque = np.asarray(ideal_torque, dtype=np.float64)
        if ideal_torque.shape != self.state.shape:
            raise ValueError(
                "ideal torque shape {} does not match {} joints".format(
                    ideal_torque.shape, len(self.state)
                )
            )
        if not self.enabled:
            return ideal_torque.copy()

        self.buffer[:, 1:] = self.buffer[:, :-1]
        self.buffer[:, 0] = ideal_torque
        joint_index = np.arange(len(self.state))
        lower = self.buffer[joint_index, self.delay_floor]
        upper = self.buffer[joint_index, self.delay_floor + 1]
        delayed = (1.0 - self.delay_fraction) * lower + self.delay_fraction * upper

        output = ideal_torque.copy()
        static_mask = self.model_codes == MODEL_STATIC_GAIN
        output[static_mask] = self.gain[static_mask] * ideal_torque[static_mask]

        fopdt_mask = self.model_codes == MODEL_FOPDT
        self.state[fopdt_mask] += self.alpha[fopdt_mask] * (
            self.gain[fopdt_mask] * delayed[fopdt_mask] - self.state[fopdt_mask]
        )
        output[fopdt_mask] = self.state[fopdt_mask]
        return output
