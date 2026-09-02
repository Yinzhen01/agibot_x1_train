import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "humanoid"
    / "utils"
    / "actuator_torque_dynamics.py"
)
SPEC = importlib.util.spec_from_file_location("actuator_torque_dynamics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def class_literal_assignments(path, outer_name, nested_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    outer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == outer_name)
    nested = next(node for node in outer.body if isinstance(node, ast.ClassDef) and node.name == nested_name)
    result = {}
    for node in nested.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


class ActuatorTorqueDynamicsTest(unittest.TestCase):
    def test_static_gain_is_immediate_and_joint_specific(self):
        model = MODULE.NumpyActuatorTorqueDynamics(
            ["yaw", "knee"],
            {"yaw": {"model": "static_gain", "gain": 0.52}},
            0.001,
        )
        actual = model.update(np.array([10.0, 10.0]))
        np.testing.assert_allclose(actual, [5.2, 10.0], rtol=0.0, atol=1e-12)

    def test_fopdt_delays_filters_and_converges_to_gain(self):
        model = MODULE.NumpyActuatorTorqueDynamics(
            ["pitch"],
            {
                "pitch": {
                    "model": "fopdt",
                    "delay_ms": 2.5,
                    "time_constant_ms": 2.0,
                    "gain": 0.8,
                }
            },
            0.001,
        )
        response = np.array([model.update(np.array([10.0]))[0] for _ in range(100)])
        self.assertEqual(response[0], 0.0)
        self.assertEqual(response[1], 0.0)
        self.assertGreater(response[2], 0.0)
        self.assertLess(response[2], 8.0)
        self.assertAlmostEqual(response[-1], 8.0, places=8)

        model.reset()
        self.assertEqual(model.update(np.array([0.0]))[0], 0.0)

    def test_unknown_joint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown joints"):
            MODULE.resolve_actuator_torque_dynamics(
                ["known"],
                {"missing": {"model": "static_gain", "gain": 1.0}},
                0.001,
            )

    def test_x1_config_contains_four_fopdt_and_two_static_hip_models(self):
        config_path = MODULE_PATH.parents[1] / "envs" / "x1" / "x1_dh_stand_config.py"
        control = class_literal_assignments(config_path, "X1DHStandCfg", "control")
        specs = control["actuator_torque_dynamics"]
        models = [spec["model"] for spec in specs.values()]
        self.assertTrue(control["use_actuator_torque_dynamics"])
        self.assertEqual(len(specs), 6)
        self.assertEqual(models.count("fopdt"), 4)
        self.assertEqual(models.count("static_gain"), 2)
        self.assertTrue(all("hip_" in name for name in specs))


if __name__ == "__main__":
    unittest.main()
