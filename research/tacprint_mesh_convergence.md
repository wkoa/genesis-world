# TacPrint three-level mesh audit

2026-11-01 (study started 2026-10-31). **Spatial convergence is not accepted. Strict nonpenetration is not accepted.**

This audit follows the opt-in convergence correction in 7d762fab (1.4.5+simulation.2). It does not modify constitutive derivatives or implement CCD/monolithic nonlinear coupling.

The same piecewise-planar domain is refined by conforming longest-edge-star bisection. New support nodes inherit constrained parent edges. Unlike interior centroid enrichment, both volume and contact-face resolution change. The measured volume is unchanged to floating-point accuracy.

| Mesh | Nodes / tetrahedra | Volume-weighted RMS / maximum edge mm | Steps | Ideal-sphere penetration mm | Largest measured full-step relative force residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| mesh-coarse | 3038 / 9674 | 4.590 / 16.285 | 60/60 | 0.12968 | 45.707% |
| mesh-edge1 | 6295 / 31768 | 2.476 / 11.592 | 60/60 | 0.09779 | 53.179% |
| mesh-edge2 | 20654 / 103279 | 1.502 / 8.096 | 48/60 | 0.07839 | 64.025% |

All maxima in an incomplete row refer only to its retained prefix. Native macOS arm64 CPU FP64, four threads per process; dt=5ms, 0.3s horizon; E=100kPa, nu=0.45, density=1000kg/m^3. Hydroelastic modulus=1e6Pa and SAP gradient scale=1e12Pa/m. A force-limited PD slider drives the same sphere. FEM budget: 20 Newton/1000 CG; SAP budget: 30 Newton/1000 CG. Other tolerances follow the opt-in FEM-tight policy. No runtime performance comparison is made.

Reference maximum sampled depths: 1.459738 / 1.459738 / 1.460920mm. Minimum element mean-ratio qualities: 0.001672 / 0.001788 / 0.001320 (regular tetrahedron=1). These are diagnostics, not a demonstrated unique cause.

This uniform 5ms/1e6Pa study differs from the previous 1ms/1e7Pa low-penetration configuration. It does not establish convergence of that other configuration.

## Matched-travel diagnostics

Linear interpolation on the first loading branch at 1.2, 1.5 and 1.75mm actual travel. Coarse P1 fields are prolonged exactly; the full reference-domain L2 difference is integrated on fine tetrahedra, and Linf is measured on all fine nodes. Forces are native contact impulse/dt; energy uses the polynomial SNH density. Missing metrics remain unavailable. Matching travel alone does not equate dynamic histories.

| Adjacent levels | Relative L2 | Linf um | Relative reaction | Relative energy |
| --- | ---: | ---: | ---: | ---: |
| mesh-coarse → mesh-edge1 | 7.752% | 60.852 | 44.735% | 42.339% |
| mesh-edge1 → mesh-edge2 | 15.991% | 53.661 | 41.029% | 44.228% |

Within the common recorded interval, maximum same-time travel difference: 50.005um. Maximum same-time velocity difference: 0.855mm/s.

## Gate

The example mesh_convergence.py requires three genuine h levels, common domain/support/configuration/loading history, complete integration, converged subproblems, full-step relative force residual <=1e-3, decreasing adjacent errors, finest relative field/reaction/energy differences <=2%, and Linf <=5um. The load-history policy is <=1um travel and <=0.1mm/s velocity difference. These are stated engineering tolerances, not sensor calibration. No spatial pass can certify collision freedom.

- `finest_displacement_max_m`
- `finest_displacement_relative_l2`
- `finest_energy_relative`
- `finest_reaction_relative`
- `load_path_matched_missing`
- `mesh-coarse:full_step_force_residual`
- `mesh-edge1:full_step_force_residual`
- `mesh-edge2:full_step_force_residual`
- `mesh-edge2:integration_incomplete`
- `nondecreasing_displacement_relative_l2`
- `nondecreasing_energy_relative`

These failures remain unresolved. In particular, individual FEM/SAP convergence does not imply a balanced full nonlinear step. Larger contact penalties previously inverted elements. Next investigate consistent nonlinear coupling and a prescribed, identical physical loading path before interpreting spatial errors; nonpenetration separately needs feasible geometry, CCD and feasible-step/barrier handling.

## Reuse

```bash
python examples/deformable/mesh_convergence.py --input research/tacprint_mesh_convergence.json --output /tmp/acceptance.json
python -m unittest tests.deformable.test_mesh_convergence
```

The input manifest contains measured mesh levels, adjacent field/force/energy errors and explicitly verified prerequisites; exit code 2 means rejected evidence. Unit fixtures only test policy behavior. Private CAD, trajectories and local paths are not distributed. No registered Studio runtime was upgraded. Linux/Windows and accelerator paths were not validated in this audit.
