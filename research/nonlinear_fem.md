# Nonlinear elasticity in frozen-contact coupling

Experimental source version 1.4.5+simulation.3. Disabled by default through
`SAPCouplerOptions.nonlinear_fem=False`.

When enabled, the contact iteration reevaluates the material energy, force and tangent
at x=x_n+dt*v. The FEM velocity gradient is -dt times the native inertial/elastic force.
The existing matrix-vector product and PCG compute an update; actual energy backtracking
replaces the quadratic line model. Non-descent updates use a mass-scaled negative gradient.
Nonfinite and nonpositive-J candidates are rejected. The final gradient and contact impulses
are reassembled even at an iteration limit.

The contact configuration, regularization and rigid mass remain frozen. This is not
continuous collision detection or a complete geometrically nonlinear contact solve.
Only single-environment/single-substep isolated rigid-FEM scenes with zero FEM damping
are accepted. Polynomial Stable Neo-Hookean elasticity has been tested; other materials,
frictional/rotating scenes, moving supports and other backends remain unverified.

On one uncalibrated 3,038-node/9,674-tetrahedron fixture, 60 steps at 5 ms:

| Metric | Frozen elasticity | Updated elasticity |
|---|---:|---:|
| Maximum relative free nodal force residual | 0.451821 | 2.42577e-5 |
| Terminal force residual (N) | 0.0300631 | 2.01549e-8 |
| Maximum mechanical penetration metric (mm) | 0.141193 | 0.140441 |

Raw native nodal impulses and FP64 states reproduce the balance diagnostic offline.
The nonpenetration requirement still fails; mesh convergence and physical calibration
are not established. CPU timing was not isolated and is not a speed comparison.

`tests.deformable.test_nonlinear_coupling` checks the compiled objective derivative,
candidate rejection/restoration and unsupported configurations. Existing material and
solver diagnostic checks remain in place. With the feature disabled, 61 historical display
states match bitwise and the saved FP64 joint positions match exactly; historical FP64
body states were unavailable for that comparison.
