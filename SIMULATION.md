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
