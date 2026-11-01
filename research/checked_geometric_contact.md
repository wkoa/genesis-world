# Checked geometric-contact research interface

The default contact law is unchanged. Source version: 1.4.5+simulation.4.
This revision adds a host-backend interface for the nonlinear FEM objective,
its matrix-free velocity Hessian product, and a checked step transaction.
The current experimental host Newton/PCG and geometric-potential adapter live
in the Studio simulation research scripts, not in the installed Genesis package.

`SAPCoupler.set_geometric_contact_solver(backend)` requires `nonlinear_fem`.
The backend implements `solve(f, coupler)`, `stats()` and `reset()`. It replaces
the frozen contact projection during coupling, including steps with no original
contact slots. Call through `CheckedStep.run(command, validate)`; a backend
cannot advance through an unchecked coupling step. No backend is auto-attached.

`CheckedStep` is limited to CPU FP64, one environment/substep, no gradients,
callbacks, sensors, emitters, cameras, viewer or recorders. Commands may mutate
only checkpointed scene inputs; external controller state stays immutable until
acceptance. A rejected candidate or raised validator error restores solver
INFO/STATE/DERIVED arrays, control inputs, simulator steps, substep cursor and
CPU error flags. Existing errors are not cleared. Coupler contact slots and
scratch impulses/diagnostics are invalidated on restore, not exposed as forces
from an accepted interval. Nested transactions are rejected. Wall-clock timing
and external audit logs intentionally retain the failed attempt.

This is an in-process state transaction, not serialization of a runtime backend.
Full scene export/checkpoint and serialized scene restore reject an attached
research backend rather than silently omitting it. The backend must not keep
uncheckpointed physical history; the current adapter rebuilds contact data.
There is no automatic step-size change or geometry projection on failure.
The host adapter starts Newton from a CCD-limited native free prediction; scaling
this initializer does not change the time step or commit a clipped physical state.

The validated adapter scope is one normal translating head, fixed supports,
zero friction/damping/gravity and polynomial Stable Neo-Hookean elasticity.
The final isolated adapter applies its normal potential to head/body pairs only.
Body-self CCD remains active, but a body-self reaction law is not supplied: a
motion requiring it must fail the feasible solve rather than acquire an invented
reaction. This avoids unloaded prestress from neighboring parts of one body.
It rebuilds a normal geometric barrier and its derivatives per candidate, uses
native elasticity and a shared energy line search, limits motion with CCD and
positive-J checks, and assembles interval impulses from the accepted potential
gradient. The separate trajectory check reads actual head/body vertices.
Both CCD paths use the same pinned open-source geometry library; they are
independent data paths, not different numerical algorithms.

`FEMOptions.split_surface_tets` defaults to True, preserving historical mesh
preparation. Disabling it preserves the supplied tetrahedra for a contact model
that does not require centroid enrichment. Pressure initialization still needs
valid interior vertices. In the isolated Studio fixture, reversing verified
four-child centroid splits preserves the exact polyhedral boundary and volume,
reducing 3,038 nodes/9,674 tets to 827/3,041. This is not a convergence claim.

Real fixture checks include full 60-step press/hold paths and injected rejection
in contact, with bitwise restored arrays and identical repeated candidates.
Closed-form nodal force balance and actual rigid momentum are checked separately.
Native regression: 22 tests pass, including rollback, error flags, nesting,
export rejection, constitutive derivatives and existing solver diagnostics.
Default-path FP64 body trajectories match the recorded prior run bitwise.

Mesh/time convergence and physical calibration remain separate gates. The
current research contact potential is unweighted, has an explicit activation
distance and coefficient, and is not claimed mesh independent. General rigid
rotation, friction, moving supports, batches, multiple substeps, GPU and other
platforms remain unverified. No formal runtime activation is implied.
