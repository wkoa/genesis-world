"""Compiled nonlinear energy/gradient consistency on a small disposable fixture."""

import unittest
from types import SimpleNamespace

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.couplers.nonlinear_fem import NonlinearFEMMixin


@qd.kernel
def perturb(
    coupler: qd.template(),
    fem_direction: qd.types.ndarray(),
    rigid_direction: qd.types.ndarray(),
    scale: qd.f64,
):
    for v in range(coupler.fem_solver.n_vertices):
        for j in qd.static(range(3)):
            coupler.fem_state_v.v[0, v][j] += scale * fem_direction[v, j]
    for d in range(coupler.rigid_solver.n_dofs):
        coupler.rigid_state_dof.v[0, d] += scale * rigid_direction[d]


class NonlinearCouplingTests(unittest.TestCase):
    def test_option_is_disabled_by_default(self):
        self.assertFalse(gs.options.SAPCouplerOptions().nonlinear_fem)
        with self.assertRaises(gs.GenesisException):
            gs.options.SAPCouplerOptions(nonlinear_backtracking_iterations=0)

    def test_unsupported_dimensions_and_damping_are_rejected(self):
        for batches, substeps, damping in [(2, 1, 0), (1, 2, 0), (1, 1, 0.1)]:
            owner = SimpleNamespace(
                _B=batches,
                sim=SimpleNamespace(_substeps=substeps),
                fem_solver=SimpleNamespace(
                    is_active=True, _damping_alpha=damping, _damping_beta=0
                ),
                rigid_solver=SimpleNamespace(is_active=True),
            )
            with self.assertRaises(gs.GenesisException):
                NonlinearFEMMixin._build_nonlinear_fem(owner)

    def test_compiled_objective_directional_derivative(self):
        gs.init(backend=gs.cpu, precision="64", seed=0, logging_level="warning")
        try:
            scene = gs.Scene(
                show_viewer=False,
                sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)),
                fem_options=gs.options.FEMOptions(
                    use_implicit_solver=True,
                    enable_vertex_constraints=True,
                    damping_alpha=0,
                    damping_beta=0,
                    n_newton_iterations=20,
                    n_pcg_iterations=100,
                    n_linesearch_iterations=10,
                    pcg_threshold=1e-20,
                    newton_force_atol=1e-10,
                    newton_force_rtol=1e-8,
                ),
                coupler_options=gs.options.SAPCouplerOptions(
                    nonlinear_fem=True,
                    n_sap_iterations=30,
                    n_pcg_iterations=100,
                    pcg_threshold=1e-22,
                    pcg_relative_tolerance=1e-8,
                    sap_convergence_atol=1e-16,
                    sap_convergence_rtol=1e-10,
                    fem_floor_contact_type="none",
                    rigid_floor_contact_type="none",
                    rigid_rigid_contact_type="none",
                    enable_fem_self_tet_contact=False,
                ),
            )
            body = scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(0, 0, 0.002), size=(0.01, 0.01, 0.004), maxvolume=2e-8
                ),
                material=gs.materials.FEM.Elastic(
                    model="stable_neohookean",
                    E=100000,
                    nu=0.3,
                    rho=1000,
                    hydroelastic_modulus=1e6,
                    friction_mu=0,
                ),
            )
            xml = '<mujoco><worldbody><body pos="0 0 .0071"><body name="tip"><joint type="slide" axis="0 0 -1"/><geom type="sphere" size=".003" density="600"/></body></body></worldbody></mujoco>'
            head = scene.add_entity(
                morph=gs.morphs.MJCF(file=xml),
                material=gs.materials.Rigid(friction=0.01),
            )
            scene.build()
            initial = body.get_state().pos.detach().cpu().numpy().reshape(-1, 3)
            fixed = np.flatnonzero(
                np.isclose(initial[:, 2], initial[:, 2].min(), atol=1e-10)
            )
            body.set_vertex_constraints(fixed)
            head.set_dofs_armature(0.1)
            for _ in range(20):
                head.control_dofs_force(0.2)
                scene.step()
            coupler = scene.sim.coupler
            self.assertTrue(coupler.has_contact)
            # Disposable probe: same frozen contact configuration, a perturbed nonlinear candidate.
            rng = np.random.default_rng(0)
            direction = rng.normal(size=initial.shape) * 0.01
            direction[fixed] = 0
            rigid = np.array([0.01])
            f = 0
            perturb(coupler, direction, rigid, 0.3)
            coupler._nl_refresh_gradient(f)
            g = coupler.fem_state_v.gradient.to_numpy()[0]
            gr = coupler.rigid_state_dof.gradient.to_numpy()[0]
            analytic = float(np.sum(g * direction) + gr @ rigid)
            coupler.batch_linesearch_active.fill(True)
            epsilon = 1e-4
            values = []
            for scale in [epsilon, -2 * epsilon]:
                perturb(coupler, direction, rigid, scale)
                self.assertTrue(coupler._refresh_nonlinear_fem(f))
                coupler._nl_objective(
                    f,
                    dofs_state=scene.sim.rigid_solver.dyn_state.dofs,
                    rigid_info=scene.sim.rigid_solver.rigid_info,
                )
                values.append(float(coupler._nl_energy[0]))
            finite_difference = (values[0] - values[1]) / (2 * epsilon)
            self.assertAlmostEqual(
                finite_difference, analytic, delta=max(1e-10, abs(analytic) * 1e-5)
            )
            backup = coupler.fem_state_v.v.to_numpy().copy()
            free = np.setdiff1d(np.arange(len(initial)), fixed)
            inverted = backup.copy()
            inverted[0, free, 2] -= 100.0
            coupler.fem_state_v.v.from_numpy(inverted)
            self.assertFalse(coupler._refresh_nonlinear_fem(f))
            invalid = backup.copy()
            invalid[0, free[0], 0] = np.nan
            coupler.fem_state_v.v.from_numpy(invalid)
            self.assertFalse(coupler._refresh_nonlinear_fem(f))
            coupler.fem_state_v.v.from_numpy(backup)
            self.assertTrue(coupler._refresh_nonlinear_fem(f))
        finally:
            gs.destroy()


if __name__ == "__main__":
    unittest.main()
