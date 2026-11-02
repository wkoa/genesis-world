# Stable Neo-Hookean constitutive baseline

The `simulation` branch implements one polynomial potential for both explicit stress and implicit
energy derivatives. This is the log-free variant, not the logarithm-corrected variant used by
SuperDex. Mixing their stresses or parameter mappings changes the physical small-strain response.

For physical Lame parameters `mu_L`, `lambda_L` derived from the public `E`, `nu`, define

```
mu = mu_L
lambda_s = lambda_L + mu_L
alpha = 1 + mu / lambda_s
Psi(F) = mu/2 * (||F||^2 - 3) + lambda_s/2 * (det(F) - alpha)^2
P(F) = mu * F + lambda_s * (det(F) - alpha) * cof(F)
```

The existing implicit energy is preserved, including its irrelevant constant at identity. The
explicit stress now differentiates this same potential. At identity the stress is zero and

```
DP[I][D] = mu_L * (D + D.T) + lambda_L * trace(D) * I.
```

For any direction `D`, the exact tangent is

```
DP[F][D] = mu * D + lambda_s * (
    inner(cof(F), D) * cof(F) + (det(F) - alpha) * Dcof[F][D]
)
```

`Dcof[F][D]` is evaluated using column cross products, without `inverse(F)` or division by
`det(F)`. The energy, stress and tangent remain finite at singular and inverted configurations.
This does **not** make such configurations physically admissible or prevent element inversion.
The solver's block layout is `H[J,L][i,k] = dP[i,J]/dF[k,L]`; volume and rest-shape mapping
are applied by the FEM solver, not again by the material.

## Validation

Run with an isolated interpreter containing the Genesis runtime dependencies. Verify that the
import resolves to this checkout; a previously installed wheel does not contain the patch.

```bash
PYTHONPATH="$PWD" /path/to/python -m unittest tests.deformable.test_fem_constitutive -v
FEM_TEST_BACKEND=metal FEM_TEST_PRECISION=32 PYTHONPATH="$PWD" /path/to/python -m unittest tests.deformable.test_fem_constitutive -v
```

The standalone runner initializes CPU/float64 by default. Under the normal pytest suite it uses
the initialized backend and precision instead. It checks 80 states spanning five Poisson ratios
(-0.2, 0, 0.3, 0.49, 0.4999), compression, stretch, shear, rotations, singularity and inversion.
The actual compiled energy, stress and Hessian are compared with an independent Torch-autodiff
reference and central differences. Rest stress, small-strain moduli, rotational objectivity,
Hessian symmetry, and agreement of all material dispatch paths are checked separately.
Derivative errors are normalized by `1 + abs(mu_L) + abs(lambda_L)`.

The retained source baseline fails the derivative test at the missing Hessian. macOS arm64
CPU/float64 and Metal/float32 are the initial target checks. CUDA/Linux/Windows require native
validation; backend-independent source is not evidence that these platforms have passed.

Initial macOS arm64 validation, 2026-10-31, Python 3.13.14, Torch 2.9.0, Quadrants 1.3.3:

| Backend / precision | AD gradient error | AD Hessian error | FD gradient error | FD Hessian error |
| --- | ---: | ---: | ---: | ---: |
| CPU / float64 | 1.52e-16 | 4.09e-16 | 1.85e-11 | 1.22e-11 |
| Metal / float32 | 8.16e-8 | 2.21e-7 | 5.85e-5 | 4.86e-5 |

These are stiffness-normalized errors, not physical force errors. Near-incompressible float32
states exhibit cancellation in stresses near rest; the derivative check does not establish
quantitative accuracy for tactile calibration. The independently built research wheel also
passes the CPU/float64 derivative check. An 80-step implicit FEM/SAP diagnostic runs with this
material on 295 vertices / 960 tetrahedra, without nonfinite states or inverted elements, but
still has about 1.89 mm geometric contact overlap. It is not a non-penetration acceptance result.

## Remaining work

- The Hessian is the **exact** tangent, without a PSD projection. Large-deformation states can
  have indefinite tangents, so robust Newton/PCG behavior needs a separate stabilization design.
- This change does not add differentiable implicit time integration or implicit muscle actuation.
- SAP remains a compliant-contact approximation. Constitutive correctness does not prevent
  surface penetration, and this patch does not add CCD or an IPC barrier.
- Near-incompressible derivative checks do not prove freedom from volumetric locking.
- Tacprint geometry, material calibration, bonded layers, force accuracy and real-time performance
  remain separate validations.

The polynomial and physical-moduli mapping agree with the SNK1 specification in the sibling
`wkoa/libuipc` fork, constitution UID 10. Its runtime backend and Genesis's material implementation
remain separate; no cross-engine dynamic equivalence is claimed.


## Opt-in convergence controls (1.4.5+simulation.2)

The constitutive energy, stress and exact Hessian are unchanged. This revision adds
independent convergence controls for the free FEM prediction and the SAP velocity solve.
It does not implement PSD projection, nonlinear monolithic coupling, CCD or a no-penetration guarantee.

Both `FEMOptions` and `SAPCouplerOptions` accept `pcg_relative_tolerance` (default 0).
PCG compares the squared Euclidean residual to
`max(pcg_threshold, pcg_relative_tolerance**2 * initial_residual_squared)`.
The existing `pcg_threshold` remains a **squared** absolute tolerance, preserving default stopping.
The final converged linear residual is now recorded, rather than the previous iteration's value.
Relative mode may synchronize the device to stop fully inactive loops early; no GPU performance is asserted.

`FEMOptions.newton_force_atol` (N) and `newton_force_rtol` default to 0, disabling the new
Newton force check. When enabled, the force norm is compared to the larger absolute or relative
threshold; non-finite forces are explicitly reported. These criteria apply only to the free
prediction, before contact coupling. Legacy runs are reported as `not_monitored`, not converged.

`scene.sim.fem_solver.get_solver_stats()` and `scene.sim.coupler.get_solver_stats()` return
per-environment diagnostics. Readback synchronizes the device. Residuals are the last checked
iterate; a budget-exhausted solve remains conservatively `iteration_limit`. An unsolved/no-contact
step does not expose stale residuals. SAP diagnostics describe only the contact velocity problem.
A converged free prediction plus a converged SAP solve does not prove the full nonlinear step is balanced.

Validation in the Studio study found that the legacy SAP absolute PCG threshold prematurely
truncates millimeter-scale contact solves. Tightening it removes the extreme coarse/fine
maximum-displacement discrepancy, but common-boundary refinement still differs and penetration
remains. An independent force-balance probe also observes a substantial post-coupling residual
when both subproblems meet their own tolerances. These are research limitations, not accepted accuracy.
The disabled-default regression reproduces all 61 body frames and head coordinates of the legacy study.

CPU FP64 checks:

```bash
PYTHONPATH=. /path/to/python -m unittest tests.deformable.test_sap_convergence tests.deformable.test_fem_convergence tests.deformable.test_fem_constitutive -v
```

The compiled probes test residual-scale stopping, legacy behavior, truthful diagnostics and
non-finite force handling. The original 80-state derivative/objectivity test remains separate.
Only source-based research execution is validated here; this is not a published wheel or a
Studio registered-runtime upgrade. Linux/Windows native and accelerator paths remain unverified.


## Three-level mesh acceptance

`examples/deformable/mesh_convergence.py` evaluates measured spatial-convergence evidence
separately from contact acceptance. It requires at least three genuinely refined levels,
a common domain/support/loading path, converged subproblems and balanced full steps.
It checks decreasing field, reaction and energy differences and declared engineering tolerances.
Completion, similar maximum displacement, or an unloaded control cannot satisfy these checks.
The utility always leaves strict nonpenetration uncertified; it has no CCD evidence.

The [TacPrint audit](research/tacprint_mesh_convergence.md) records a conforming,
same-domain three-level study and its unresolved failures. No private geometry or trajectory
is distributed. This extends validation of simulation.2; it adds no new physical solver algorithm
and does not change a registered Studio runtime.

```bash
python examples/deformable/mesh_convergence.py --input STUDY.json --output acceptance.json
python -m unittest tests.deformable.test_mesh_convergence -v
```

A rejected study exits with code 2. Unit fixtures exercise policy only and are not physical data.

## Energy-only backtracking candidate (2026-11-02)

Research source 1.4.5+simulation.5 exposes `_nl_refresh_energy(f)` for the isolated
undamped single-environment/substep geometric backend. It updates candidate positions,
rejects nonpositive/nonfinite J, recomputes material energy with the existing energy
function, and evaluates the same unconstrained inertial/elastic/rigid objective.
It preserves FEM derivative/tangent scratch and restores the linesearch mask.
The caller must refresh derivatives at every next Newton iterate; energy evaluation
does not certify force equilibrium or replace CCD/line search/transaction rollback.
No default native solver or registered Studio runtime is switched.
