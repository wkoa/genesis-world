"""Software fixtures for the conservative gate, not physical validation data."""

import copy
import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "examples/deformable/mesh_convergence.py"
spec = importlib.util.spec_from_file_location("mesh_convergence", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return {
        "common_domain_verified": True,
        "common_support_verified": True,
        "identical_configuration": True,
        "load_path_matched": True,
        "levels": [
            {
                "id": str(index),
                "complete": True,
                "sap_converged": True,
                "fem_prediction_converged": True,
                "max_relative_force_residual": 1e-5,
                "min_jacobian": 0.9,
                "nodes": 10 * 8**index,
                "elements": 20 * 8**index,
                "h_rms_m": 0.001 / 2**index,
                "h_max_m": 0.002 / 2**index,
            }
            for index in range(3)
        ],
        "pairs": [
            {
                "displacement_relative_l2": 0.01 / factor,
                "displacement_max_m": 2e-6 / factor,
                "reaction_relative": 0.01 / factor,
                "energy_relative": 0.01 / factor,
            }
            for factor in (1, 2)
        ],
    }


class MeshGateTests(unittest.TestCase):
    def test_well_formed_fixture_passes_spatial_gate_but_never_certifies_contact(self):
        result = module.assess(fixture())
        self.assertTrue(result["mesh_convergence_accepted"])
        self.assertFalse(result["strict_nonpenetration_accepted"])

    def test_close_scalar_results_do_not_override_failed_prerequisites(self):
        for key in (
            "common_domain_verified",
            "common_support_verified",
            "identical_configuration",
            "load_path_matched",
        ):
            data = fixture()
            data[key] = False
            self.assertFalse(module.assess(data)["mesh_convergence_accepted"])
        for key, value in [
            ("complete", False),
            ("sap_converged", False),
            ("fem_prediction_converged", False),
            ("max_relative_force_residual", 0.1),
            ("min_jacobian", -0.1),
            ("h_rms_m", float("nan")),
        ]:
            data = fixture()
            data["levels"][1][key] = value
            self.assertFalse(module.assess(data)["mesh_convergence_accepted"])

    def test_three_levels_and_actual_refinement_are_required(self):
        data = fixture()
        data["levels"].pop()
        data["pairs"].pop()
        self.assertFalse(module.assess(data)["mesh_convergence_accepted"])
        data = fixture()
        data["levels"][2]["h_max_m"] = data["levels"][1]["h_max_m"]
        self.assertIn("no_spatial_refinement", module.assess(data)["reasons"])

    def test_bad_or_increasing_field_energy_and_reaction_errors_are_rejected(self):
        for key in fixture()["pairs"][0]:
            for value in (float("nan"), True, -0.1, 1):
                data = copy.deepcopy(fixture())
                data["pairs"][-1][key] = value
                self.assertFalse(module.assess(data)["mesh_convergence_accepted"])


if __name__ == "__main__":
    unittest.main()
