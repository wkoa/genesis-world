"""Compiled force-residual stopping checks for the unconstrained FEM prediction."""
import unittest
from types import SimpleNamespace

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.fem_solver import FEMSolver


@qd.data_oriented
class ForceProbe:
    def __init__(self):
        self._B, self.n_vertices = 2, 2
        self._newton_force_atol, self._newton_force_rtol = 1e-9, 1e-3
        self._monitor_newton, self._force_monitor_evaluated = True, True
        self.batch_active = qd.field(gs.qd_bool, shape=2)
        self.batch_active.fill(True)
        self.newton_force_sq = qd.field(gs.qd_float, shape=2)
        self.newton_initial_force_sq = qd.field(gs.qd_float, shape=2)
        self.newton_iterations = qd.field(gs.qd_int, shape=2)
        self.newton_diverged = qd.field(gs.qd_bool, shape=2)
        self.elements_v_energy = qd.types.struct(force=gs.qd_vec3).field(shape=(2, 2))

    @qd.kernel
    def set_forces(self, factor: qd.f64):
        self.elements_v_energy[0, 0].force = qd.Vector([1e-4 * factor, 0.0, 0.0])
        self.elements_v_energy[1, 0].force = qd.Vector([100.0 * factor, 0.0, 0.0])


class PolicyTests(unittest.TestCase):
    def test_legacy_mode_and_pre_step_state_are_not_reported_as_converged(self):
        owner = SimpleNamespace(_B=1, _monitor_newton=False)
        self.assertEqual(FEMSolver.get_solver_stats(owner)[0]['status'], 'not_monitored')
        owner._monitor_newton = True
        owner._force_monitor_evaluated = False
        self.assertEqual(FEMSolver.get_solver_stats(owner)[0]['status'], 'not_evaluated')
        options = gs.options.FEMOptions()
        self.assertEqual(options.newton_force_atol, 0)
        self.assertEqual(options.newton_force_rtol, 0)
        self.assertEqual(options.pcg_relative_tolerance, 0)
        with self.assertRaises(gs.GenesisException):
            gs.options.FEMOptions(newton_force_rtol=-1)


class CompiledForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gs.init(backend=gs.cpu, precision='64', logging_level='warning')

    @classmethod
    def tearDownClass(cls):
        gs.destroy()

    def test_relative_force_reduction_has_consistent_stopping_across_scales(self):
        probe = ForceProbe()
        probe.set_forces(1)
        FEMSolver.check_newton_force(probe, 0)
        np.testing.assert_array_equal(probe.batch_active.to_numpy(), [True, True])
        probe.set_forces(1e-4)
        FEMSolver.check_newton_force(probe, 1)
        np.testing.assert_array_equal(probe.batch_active.to_numpy(), [False, False])
        np.testing.assert_array_equal(probe.newton_iterations.to_numpy(), [1, 1])
        self.assertTrue(all(row['status'] == 'converged' for row in FEMSolver.get_solver_stats(probe)))

    def test_nonfinite_force_is_stopped_and_explicitly_reported(self):
        probe = ForceProbe()
        probe.set_forces(float('nan'))
        FEMSolver.check_newton_force(probe, 0)
        self.assertTrue(all(row['status'] == 'nonfinite' and row['force_residual_N'] is None
                            for row in FEMSolver.get_solver_stats(probe)))


if __name__ == '__main__':
    unittest.main()
