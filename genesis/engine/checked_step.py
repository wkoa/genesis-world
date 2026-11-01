"""Headless research step transaction: validate a candidate before publishing it.

This supplies rollback, not a contact law. Validators must check the actual path.
No projection, partial time advance, or silent retry is performed.
"""
import numpy as np

import genesis as gs


class StepRejected(RuntimeError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__(audit.get('reason', 'Candidate step rejected'))


def _arrays_equal(first, last):
    if first.keys() != last.keys():
        return False
    for key in first:
        a, b = first[key], last[key]
        if hasattr(a, 'detach'):
            a = a.detach().cpu().numpy()
        if hasattr(b, 'detach'):
            b = b.detach().cpu().numpy()
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            return False
    return True


class CheckedStep:
    """Single-environment CPU FP64 SAP transaction without external observers.

    Commands may modify only checkpointed scene state. Keep controller state
    outside the command immutable until this call returns successfully.
    Validation failures restore INFO/STATE/DERIVED arrays, controls and clocks;
    coupler scratch diagnostics are invalidated, never reused as accepted forces.
    """
    def __init__(self, scene):
        from genesis.engine.couplers.sap_coupler import SAPCoupler
        sim = scene.sim
        if (gs.backend != gs.cpu or gs.np_float != np.float64 or sim._B != 1
                or sim._substeps != 1 or sim.requires_grad
                or not isinstance(sim.coupler, SAPCoupler)):
            raise ValueError('CheckedStep requires CPU FP64, one environment/substep, SAP and no gradients')
        self.scene = scene
        self._busy = False
        self.last_audit = None
        self._check_observers()

    def _check_observers(self):
        scene = self.scene
        if (scene._pre_step_callbacks or scene._recorder_manager._recorders
                or scene.sim._sensor_manager._sensors_by_type or scene._emitters
                or scene._visualizer.cameras or scene._visualizer.viewer is not None):
            raise ValueError('CheckedStep requires no callbacks, recorders, sensors, emitters, cameras or viewer')

    def run(self, command, validate):
        if self._busy or getattr(self.scene.sim, '_checked_step_active', False):
            raise RuntimeError('Nested step transaction is unsupported')
        self._check_observers()
        sim = self.scene.sim
        sim.check_errno()
        checkpoint = sim.__getstate__()
        error_state = [(solver, solver._errno_latched.clone(), solver._errno.to_numpy().copy())
                       for solver in sim.active_solvers if solver._errno is not None]
        cursor = sim._cur_substep_global
        errno = sim._has_errno_flags
        self._busy = True
        sim._checked_step_active = True
        self.last_audit = None
        try:
            command()
            self.scene.step(update_visualizer=False, refresh_visualizer=False)
            sim.check_errno()
            audit = validate()
            if not isinstance(audit, dict) or type(audit.get('accepted')) is not bool:
                raise ValueError('Validator must return an explicit accepted boolean')
            self.last_audit = dict(audit, rolled_back=False)
            if not audit['accepted']:
                raise StepRejected(audit)
            return audit
        except BaseException as error:
            # Restore before reporting any candidate/validation error, including interrupts.
            sim.__setstate__(checkpoint)
            sim._cur_substep_global = cursor
            sim._has_errno_flags = errno
            for solver, latched, raw in error_state:
                solver._errno_latched.copy_(latched)
                solver._errno.from_numpy(raw)
            restored = sim.__getstate__()
            equal = _arrays_equal({'steps': checkpoint.steps}, {'steps': restored.steps})
            equal = equal and checkpoint.solvers.keys() == restored.solvers.keys()
            for name, state in checkpoint.solvers.items():
                equal = equal and _arrays_equal(state.arrays, restored.solvers[name].arrays)
            self.last_audit = dict(self.last_audit or {'accepted': False, 'reason': f'{type(error).__name__}: {error}'}, rolled_back=True,
                                   state_arrays_bitwise_restored=equal,
                                   contact_diagnostics='invalidated')
            if not equal:
                raise RuntimeError('Step rollback failed state verification; scene must not advance')
            raise
        finally:
            sim._checked_step_active = False
            self._busy = False
