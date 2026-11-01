"""Experimental nonlinear elasticity inside a frozen-contact velocity solve.

The contact configuration, regularization and rigid mass matrix remain frozen.
This is not a continuous collision or fully geometric nonlinear contact solver.
"""

import math

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils import array_class


@qd.data_oriented
class NonlinearFEMMixin:
    def _build_nonlinear_fem(self):
        fem = self.fem_solver
        if (
            self._B != 1
            or self.sim._substeps != 1
            or not fem.is_active
            or not self.rigid_solver.is_active
        ):
            gs.raise_exception(
                "nonlinear_fem requires one environment/substep and active rigid/FEM solvers"
            )
        if fem._damping_alpha != 0 or fem._damping_beta != 0:
            gs.raise_exception("nonlinear_fem currently requires zero FEM damping")
        if (
            self.rigid_solver.n_equalities
            or len(self.contact_handlers) != 1
            or not self._enable_rigid_fem_contact
        ):
            gs.raise_exception(
                "nonlinear_fem currently supports isolated rigid-FEM contact only"
            )
        self._nl_base_fem = qd.Vector.field(3, gs.qd_float, shape=(1, fem.n_vertices))
        self._nl_base_rigid = qd.field(gs.qd_float, shape=(1, self.rigid_solver.n_dofs))
        self._nl_energy = qd.field(gs.qd_float, shape=(1,))
        self._nl_min_j = qd.field(gs.qd_float, shape=())
        self._nl_slope = qd.field(gs.qd_float, shape=())
        self._nl_status = "not_evaluated"
        self._nl_iterations = 0
        self._nl_backtracks = 0

    def _refresh_nonlinear_fem(self, f):
        self._nl_update_positions(f)
        if not np.isfinite(self._nl_min_j[None]) or self._nl_min_j[None] <= 0:
            return False
        # Preserve the free-prediction diagnostic flags; only its scratch force/tangent changes.
        fem = self.fem_solver
        flags = fem.batch_active.to_numpy().copy()
        fem.batch_active.fill(True)
        try:
            fem.compute_ele_hessian_gradient(f)
            fem.accumulate_vertex_force_preconditioner(f)
        finally:
            fem.batch_active.from_numpy(flags)
        return True

    def _nl_refresh_gradient(self, f):
        self.batch_active.fill(True)
        if not self._refresh_nonlinear_fem(f):
            gs.raise_exception(
                "nonlinear_fem received an inverted or nonfinite candidate"
            )
        self._nl_fem_gradient()
        self.init_rigid_unconstrained_gradient(
            dofs_state=self.rigid_solver.dyn_state.dofs
        )
        self.compute_rigid_unconstrained_gradient(
            rigid_info=self.rigid_solver.rigid_info
        )
        self.compute_constraint_contact_gradient_hessian_diag_prec()
        self.check_sap_convergence(rigid_info=self.rigid_solver.rigid_info)
        if not np.isfinite(self.sap_state.to_numpy()["gradient_norm"]).all():
            gs.raise_exception("nonlinear_fem produced a nonfinite coupled gradient")

    def _solve_nonlinear_fem(self, f):
        self.pcg_state.fill(0)
        self._init_sap_solve(f, dofs_state=self.rigid_solver.dyn_state.dofs)
        self._nl_status = "iteration_limit"
        self._nl_backtracks = 0
        for iteration in range(self._n_sap_iterations):
            self._nl_iterations = iteration
            self._nl_refresh_gradient(f)
            if not self.batch_active.to_numpy().any():
                self._nl_status = "converged"
                break
            self.pcg_solve()
            self._nl_save_base_and_slope(False, rigid_info=self.rigid_solver.rigid_info)
            slope = float(self._nl_slope[None])
            if not math.isfinite(slope) or slope >= 0:
                self._nl_save_base_and_slope(
                    True, rigid_info=self.rigid_solver.rigid_info
                )
                slope = float(self._nl_slope[None])
            self.batch_linesearch_active.fill(True)
            self._nl_objective(
                f,
                dofs_state=self.rigid_solver.dyn_state.dofs,
                rigid_info=self.rigid_solver.rigid_info,
            )
            before = float(self._nl_energy[0])
            accepted = False
            alpha = 1.0
            if math.isfinite(before) and math.isfinite(slope) and slope < 0:
                for _ in range(self.options.nonlinear_backtracking_iterations):
                    self._nl_trial(alpha)
                    feasible = self._refresh_nonlinear_fem(f)
                    if feasible:
                        self._nl_objective(
                            f,
                            dofs_state=self.rigid_solver.dyn_state.dofs,
                            rigid_info=self.rigid_solver.rigid_info,
                        )
                        energy = float(self._nl_energy[0])
                        if (
                            math.isfinite(energy)
                            and energy <= before + 1e-4 * alpha * slope
                        ):
                            accepted = True
                            break
                    alpha *= 0.5
                    self._nl_backtracks += 1
            if not accepted:
                self._nl_trial(0.0)
                self._nl_status = "line_search_failed"
                break
        # Reassemble gamma and both impulses at the actual returned iterate, including budget exits.
        self._nl_refresh_gradient(f)
        if not self.batch_active.to_numpy().any():
            self._nl_status = "converged"

    @qd.kernel
    def _nl_update_positions(self, f: qd.i32):
        self._nl_min_j[None] = np.inf
        for b, v in qd.ndrange(1, self.fem_solver.n_vertices):
            self.fem_solver.elements_v[f + 1, v, b].pos = (
                self.fem_solver.elements_v[f, v, b].pos
                + self.sim._substep_dt * self.fem_state_v.v[b, v]
            )
        for e in range(self.fem_solver.n_elements):
            j, _ = self.fem_solver._compute_ele_J_F(f + 1, e, 0)
            if qd.math.isnan(j) or qd.math.isinf(j):
                qd.atomic_min(self._nl_min_j[None], -np.inf)
            else:
                qd.atomic_min(self._nl_min_j[None], j)

    @qd.kernel
    def _nl_fem_gradient(self):
        dt = self.sim._substep_dt
        for v in range(self.fem_solver.n_vertices):
            self.fem_state_v.gradient[0, v] = (
                -dt * self.fem_solver.elements_v_energy[0, v].force
            )
            self.pcg_fem_state_v.diag3x3[0, v] = (
                self.fem_solver.pcg_state_v.diag3x3[0, v] * dt**2
            )
            if qd.static(self.fem_solver._enable_vertex_constraints):
                if self.fem_solver.vertex_constraints.is_constrained[v, 0]:
                    self.fem_state_v.gradient[0, v].fill(0.0)

    @qd.kernel
    def _nl_save_base_and_slope(
        self, fallback: qd.i32, rigid_info: array_class.RigidInfo
    ):
        self._nl_slope[None] = 0.0
        for v in range(self.fem_solver.n_vertices):
            self._nl_base_fem[0, v] = self.fem_state_v.v[0, v]
            if fallback:
                self.pcg_fem_state_v.x[0, v] = (
                    -self.fem_state_v.gradient[0, v]
                    / self.fem_solver.elements_v_info[v].mass
                )
            qd.atomic_add(
                self._nl_slope[None],
                self.fem_state_v.gradient[0, v].dot(self.pcg_fem_state_v.x[0, v]),
            )
        for d in range(self.rigid_solver.n_dofs):
            self._nl_base_rigid[0, d] = self.rigid_state_dof.v[0, d]
            if fallback:
                self.pcg_rigid_state_dof.x[0, d] = (
                    -self.rigid_state_dof.gradient[0, d] / rigid_info.mass_mat[d, d, 0]
                )
            qd.atomic_add(
                self._nl_slope[None],
                self.rigid_state_dof.gradient[0, d] * self.pcg_rigid_state_dof.x[0, d],
            )

    @qd.kernel
    def _nl_trial(self, alpha: qd.f64):
        for v in range(self.fem_solver.n_vertices):
            self.fem_state_v.v[0, v] = (
                self._nl_base_fem[0, v] + alpha * self.pcg_fem_state_v.x[0, v]
            )
        for d in range(self.rigid_solver.n_dofs):
            self.rigid_state_dof.v[0, d] = (
                self._nl_base_rigid[0, d] + alpha * self.pcg_rigid_state_dof.x[0, d]
            )

    @qd.kernel
    def _nl_objective(
        self,
        f: qd.i32,
        dofs_state: array_class.DofsState,
        rigid_info: array_class.RigidInfo,
    ):
        self._nl_energy.fill(0.0)
        for v in range(self.fem_solver.n_vertices):
            delta = (
                self.fem_solver.elements_v[f + 1, v, 0].pos
                - self.fem_solver.elements_v_energy[0, v].inertia
            )
            qd.atomic_add(
                self._nl_energy[0],
                0.5
                * self.fem_solver.elements_v_info[v].mass_over_dt2
                * delta.norm_sqr(),
            )
        for e in range(self.fem_solver.n_elements):
            qd.atomic_add(
                self._nl_energy[0],
                self.fem_solver.elements_i[e].V
                * self.fem_solver.elements_el_energy[0, e].energy,
            )
        self.compute_rigid_energy(
            self._nl_energy, dofs_state=dofs_state, rigid_info=rigid_info
        )
        for contact in qd.static(self.contact_handlers):
            contact.compute_energy(self._nl_energy)
