"""Compiled PCG stopping policy and truthful SAP diagnostics (CPU FP64)."""
import unittest

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.couplers.sap_coupler import SAPCoupler


@qd.data_oriented
class PcgProbe:
    def __init__(self, relative):
        self._B = 2
        self._pcg_threshold = 1e-30 if relative else 1e-6
        self._pcg_relative_tolerance = relative
        self.batch_pcg_active = qd.field(gs.qd_bool, shape=2)
        state = qd.types.struct(initial_rTr=gs.qd_float, residual_threshold=gs.qd_float,
            rTr=gs.qd_float, rTr_new=gs.qd_float, rTz=gs.qd_float, rTz_new=gs.qd_float, beta=gs.qd_float)
        self.pcg_state = state.field(shape=2)

    @qd.kernel
    def initialize(self):
        for i in range(2):
            self.batch_pcg_active[i] = True
            self.pcg_state[i].rTr = 1e-12 if i == 0 else 100.0
            self.pcg_state[i].rTz = 1.0
        SAPCoupler.init_pcg_active(self)

    @qd.kernel
    def converge(self):
        for i in range(2):
            self.pcg_state[i].rTr_new = self.pcg_state[i].residual_threshold * .5
            self.pcg_state[i].rTz_new = .5
        SAPCoupler.check_pcg_convergence(self)


class FakeField:
    def __init__(self, value):
        self.value = value

    def to_numpy(self):
        return self.value


class SapDiagnosticsTests(unittest.TestCase):
    def test_relative_tolerance_is_opt_in_and_nonnegative(self):
        self.assertEqual(gs.options.SAPCouplerOptions().pcg_relative_tolerance, 0)
        with self.assertRaises(gs.GenesisException):
            gs.options.SAPCouplerOptions(pcg_relative_tolerance=-1)

    def test_before_first_step_is_not_evaluated(self):
        owner = type('Owner', (), {'_B': 1})()
        self.assertEqual(SAPCoupler.get_solver_stats(owner)[0]['status'], 'not_evaluated')

    def test_no_contact_does_not_expose_previous_step(self):
        owner = type('Owner', (), {'has_contact': False, '_B': 2})()
        rows = SAPCoupler.get_solver_stats(owner)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['status'] == 'no_contact' and row['pcg_residual_sq'] is None for row in rows))

    def test_iteration_limit_and_nonfinite_are_not_success(self):
        owner = type('Owner', (), {})()
        owner.has_contact, owner._B = True, 3
        owner._sap_convergence_atol, owner._sap_convergence_rtol = 1e-12, 1e-8
        owner.batch_active = FakeField(np.array([False, True, False]))
        owner.sap_state = FakeField({'gradient_norm': np.array([0., 1., np.nan]),
            'momentum_norm': np.zeros(3), 'impulse_norm': np.zeros(3)})
        owner.pcg_state = FakeField({'rTr': np.zeros(3), 'initial_rTr': np.ones(3),
            'residual_threshold': np.full(3, 1e-12), 'iterations': np.array([1, 500, 0])})
        rows = SAPCoupler.get_solver_stats(owner)
        self.assertEqual([r['status'] for r in rows], ['converged', 'iteration_limit', 'nonfinite'])
        self.assertIsNone(rows[2]['weighted_gradient_norm_sq'])
        self.assertEqual(rows[1]['last_pcg_iterations'], 500)


class CompiledPcgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gs.init(backend=gs.cpu, precision='64', logging_level='warning')

    @classmethod
    def tearDownClass(cls):
        gs.destroy()

    def test_legacy_absolute_stopping_is_preserved(self):
        probe = PcgProbe(0)
        probe.initialize()
        np.testing.assert_array_equal(probe.batch_pcg_active.to_numpy(), [False, True])

    def test_relative_reduction_is_consistent_across_residual_scales(self):
        probe = PcgProbe(1e-3)
        probe.initialize()
        np.testing.assert_array_equal(probe.batch_pcg_active.to_numpy(), [True, True])
        before = probe.pcg_state.to_numpy()
        np.testing.assert_allclose(before['residual_threshold'] / before['initial_rTr'], [1e-6, 1e-6])
        probe.converge()
        np.testing.assert_array_equal(probe.batch_pcg_active.to_numpy(), [False, False])
        after = probe.pcg_state.to_numpy()
        np.testing.assert_array_equal(after['rTr'], after['rTr_new'])


if __name__ == '__main__':
    unittest.main()
