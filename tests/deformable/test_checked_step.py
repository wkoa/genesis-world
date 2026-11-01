"""Actual CPU FP64 state/control/clock rollback, including failed validators."""
import unittest
import numpy as np
import genesis as gs
from genesis.engine.checked_step import CheckedStep, StepRejected, _arrays_equal


class CheckedStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gs.init(backend=gs.cpu, precision='64', seed=0, logging_level='warning')
        scene = gs.Scene(show_viewer=False,
            sim_options=gs.options.SimOptions(dt=.001, gravity=(0, 0, 0)),
            fem_options=gs.options.FEMOptions(use_implicit_solver=True,
                enable_vertex_constraints=True, damping_alpha=0, damping_beta=0),
            coupler_options=gs.options.SAPCouplerOptions(
                fem_floor_contact_type='none', rigid_floor_contact_type='none',
                rigid_rigid_contact_type='none', enable_fem_self_tet_contact=False))
        cls.body = scene.add_entity(morph=gs.morphs.Box(pos=(0, 0, .002),
            size=(.01, .01, .004), maxvolume=2e-8),
            material=gs.materials.FEM.Elastic(model='stable_neohookean',
                E=100000, nu=.3, rho=1000, friction_mu=0))
        cls.head = scene.add_entity(morph=gs.morphs.MJCF(file='<mujoco><worldbody><body pos="0 0 .008"><body name="tip"><joint type="slide" axis="0 0 -1"/><geom type="sphere" size=".003" density="600"/></body></body></worldbody></mujoco>'),
            material=gs.materials.Rigid(friction=.01))
        scene.build()
        initial = cls.body.get_state().pos.detach().cpu().numpy().reshape(-1, 3)
        cls.body.set_vertex_constraints(np.flatnonzero(np.isclose(initial[:, 2], initial[:, 2].min())))
        cls.head.set_dofs_armature(.1)
        cls.scene, cls.initial = scene, scene.sim.__getstate__()

    @classmethod
    def tearDownClass(cls):
        gs.destroy()

    def setUp(self):
        self.scene.sim.__setstate__(self.initial)
        self.scene.sim._cur_substep_global = 0
        self.transaction = CheckedStep(self.scene)

    def command(self):
        self.head.control_dofs_force(.2)

    def test_rejection_restores_arrays_controls_and_clocks(self):
        self.transaction.run(self.command, lambda: {'accepted': True})
        before, cursor = self.scene.sim.__getstate__(), self.scene.sim._cur_substep_global
        with self.assertRaises(StepRejected):
            self.transaction.run(self.command, lambda: {'accepted': False, 'reason': 'collision'})
        after = self.scene.sim.__getstate__()
        self.assertTrue(_arrays_equal({'steps': before.steps}, {'steps': after.steps}))
        for name, state in before.solvers.items():
            self.assertTrue(_arrays_equal(state.arrays, after.solvers[name].arrays))
        self.assertEqual(cursor, self.scene.sim._cur_substep_global)
        self.assertTrue(self.transaction.last_audit['state_arrays_bitwise_restored'])
        self.assertEqual(self.scene.sim.coupler.get_solver_stats()[0]['status'], 'not_evaluated')

    def test_validator_error_and_command_error_restore(self):
        def fail():
            raise ValueError('validator unavailable')
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            self.transaction.run(self.command, fail)
        self.assertTrue(self.transaction.last_audit['state_arrays_bitwise_restored'])
        def bad_command():
            self.command()
            raise RuntimeError('command interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.transaction.run(bad_command, lambda: {'accepted': True})
        self.assertTrue(self.transaction.last_audit['state_arrays_bitwise_restored'])

    def test_retry_recomputes_and_repeats_the_same_candidate(self):
        observed = []
        def reject():
            observed.append(self.body.get_state().pos.detach().cpu().numpy().copy())
            return {'accepted': False}
        with self.assertRaises(StepRejected):
            self.transaction.run(self.command, reject)
        self.transaction.run(self.command, lambda: {'accepted': True})
        np.testing.assert_array_equal(observed[0], self.body.get_state().pos.detach().cpu().numpy())
        self.assertFalse(self.transaction.last_audit['rolled_back'])

    def test_rejected_native_error_flags_are_restored(self):
        from genesis.utils.array_class import ErrorCode
        solver = self.scene.sim.rigid_solver
        def bad():
            self.command()
            solver.flag_errno(ErrorCode.INVALID_FORCE_NAN)
        with self.assertRaises(gs.GenesisException):
            self.transaction.run(bad, lambda: {'accepted': True})
        self.assertFalse(bool(solver.get_error_envs_mask().any()))
        self.scene.sim.check_errno()
        self.assertTrue(self.transaction.last_audit['state_arrays_bitwise_restored'])

    def test_preexisting_native_error_is_not_cleared(self):
        from genesis.utils.array_class import ErrorCode
        solver = self.scene.sim.rigid_solver
        solver.flag_errno(ErrorCode.INVALID_FORCE_NAN)
        try:
            with self.assertRaises(gs.GenesisException):
                self.transaction.run(self.command, lambda: {'accepted': True})
            self.assertTrue(bool(solver.get_error_envs_mask().any()))
        finally:
            solver.clear_errno()

    def test_runtime_backend_cannot_be_silently_omitted_from_export(self):
        coupler = self.scene.sim.coupler
        coupler._geometric_contact_solver = object()
        try:
            with self.assertRaisesRegex(gs.GenesisException, 'cannot be exported'):
                self.scene.__getstate__()
        finally:
            del coupler._geometric_contact_solver

    def test_transaction_marker_and_nested_calls(self):
        observed = []
        def command():
            observed.append(self.scene.sim._checked_step_active)
            with self.assertRaises(RuntimeError):
                CheckedStep(self.scene).run(self.command, lambda: {'accepted': True})
            self.command()
        self.transaction.run(command, lambda: {'accepted': True})
        self.assertEqual(observed, [True])
        self.assertFalse(self.scene.sim._checked_step_active)

    def test_backend_registration_cannot_silently_switch_physics(self):
        from types import SimpleNamespace
        from genesis.engine.couplers.sap_coupler import SAPCoupler
        def owner(steps=0):
            return SimpleNamespace(options=gs.options.SAPCouplerOptions(nonlinear_fem=True),
                                   sim=SimpleNamespace(steps=np.array([steps])))
        backend = SimpleNamespace(solve=lambda *args: None, stats=lambda: {}, reset=lambda: None)
        target = owner()
        SAPCoupler.set_geometric_contact_solver(target, backend)
        self.assertIs(target._geometric_contact_solver, backend)
        for target, value in [(target, backend), (owner(1), backend), (owner(), None), (owner(), object())]:
            with self.assertRaises(gs.GenesisException):
                SAPCoupler.set_geometric_contact_solver(target, value)

    def test_preserve_input_mesh_is_explicit(self):
        self.assertTrue(gs.options.FEMOptions().split_surface_tets)
        self.assertFalse(gs.options.FEMOptions(split_surface_tets=False).split_surface_tets)

    def test_external_callbacks_are_rejected_before_execution(self):
        self.scene._pre_step_callbacks.append(lambda: False)
        try:
            with self.assertRaisesRegex(ValueError, 'callbacks'):
                self.transaction.run(self.command, lambda: {'accepted': True})
        finally:
            self.scene._pre_step_callbacks.clear()


if __name__ == '__main__':
    unittest.main()
