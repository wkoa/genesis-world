import os
import unittest

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.utils.misc import qd_to_numpy

from tests.utils.assertions import assert_allclose


class MaterialProbe:
    # Pydantic materials are unhashable; the kernel template needs an identity-hashable owner.
    def __init__(self, material: gs.materials.FEM.Elastic):
        self.material = material


@qd.kernel
def evaluate_material(
    probe: qd.template(),
    deformation: qd.types.ndarray(),
    coefficients: qd.types.ndarray(),
    energies: qd.types.ndarray(),
    gradients: qd.types.ndarray(),
    hessian: qd.template(),
):
    for i_e in range(deformation.shape[0]):
        F = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i, j in qd.static(qd.ndrange(3, 3)):
            F[i, j] = deformation[i_e, i, j]
        mu, lam = coefficients[i_e, 0], coefficients[i_e, 1]
        J = F.determinant()
        direction = qd.Vector.zero(gs.qd_float, 3)
        energy_h, gradient_h = probe.material.compute_energy_gradient_hessian(
            mu, lam, J, F, 0.0, direction, i_e, 0, hessian
        )
        energy_g, gradient_g = probe.material.compute_energy_gradient(mu, lam, J, F, 0.0, direction, i_e, 0)
        stress = probe.material.update_stress(mu, lam, J, F, 0.0, direction)
        energies[i_e, 0] = probe.material.compute_energy(mu, lam, J, F, 0.0, direction, i_e, 0)
        energies[i_e, 1] = energy_g
        energies[i_e, 2] = energy_h
        for i, j in qd.static(qd.ndrange(3, 3)):
            gradients[i_e, 0, i, j] = stress[i, j]
            gradients[i_e, 1, i, j] = gradient_g[i, j]
            gradients[i_e, 2, i, j] = gradient_h[i, j]


class ConstitutiveTests(unittest.TestCase):
    def test_stable_neo_hookean_derivatives_and_objectivity(self):
        backend = os.environ.get("FEM_TEST_BACKEND", "cpu")
        precision = os.environ.get("FEM_TEST_PRECISION", "64")
        self.assertIn(backend, ("cpu", "metal"))
        owns_runtime = not gs._initialized
        if owns_runtime:
            gs.init(
                backend=gs.cpu if backend == "cpu" else gs.metal, precision=precision, seed=0, logging_level="warning"
            )
        backend = gs.backend.name
        precision = "64" if gs.np_float == np.float64 else "32"
        try:
            material = gs.materials.FEM.Elastic(model="stable_neohookean")
            # Singular and inverted inputs check the smooth polynomial, not admissible simulation states.
            deformations = np.stack(
                [
                    np.eye(3),
                    np.diag([0.7, 1.1, 1.2]),
                    np.diag([1.4, 0.9, 0.8]),
                    [[1, 0.8, 0.2], [0, 1, -0.3], [0, 0, 1]],
                    np.diag([0.01, 1.0, 1.0]),
                    np.diag([0, 1, 1]),
                    np.zeros((3, 3)),
                    np.diag([-0.4, 1, 1]),
                ]
            )
            rotation = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]])
            deformations = np.concatenate([deformations, rotation @ deformations])
            coefficients = []
            for nu in (-0.2, 0.0, 0.3, 0.49, 0.4999):
                elastic = gs.materials.FEM.Elastic(E=2.0, nu=nu, model="stable_neohookean")
                coefficients.extend([(elastic.mu, elastic.lam)] * len(deformations))
            deformations = np.tile(deformations, (5, 1, 1)).astype(gs.np_float)
            coefficients = np.array(coefficients, dtype=gs.np_float)
            n_cases = len(deformations)
            eps = 2e-5 if precision == "64" else 2e-3
            basis = np.eye(9, dtype=gs.np_float).reshape(9, 3, 3)
            # Evaluate actual compiled energy/gradient on both sides of each finite-difference stencil.
            deformation = np.concatenate(
                [deformations, *(deformations + sign * eps * entry for sign in (1, -1) for entry in basis)]
            )
            coefficients = np.tile(coefficients, (19, 1))
            energies = np.empty((len(deformation), 3), dtype=gs.np_float)
            gradients = np.empty((len(deformation), 3, 3, 3), dtype=gs.np_float)
            hessian = qd.Matrix.field(3, 3, dtype=gs.qd_float, shape=(1, 3, 3, len(deformation)))
            evaluate_material(MaterialProbe(material), deformation, coefficients, energies, gradients, hessian)
            tangent = qd_to_numpy(hessian, copy=True)[0].transpose(2, 3, 0, 4, 1).reshape(-1, 9, 9)
            scale = 1 + np.abs(coefficients[:n_cases]).sum(axis=1)
            tol = 1e-10 if precision == "64" else 5e-5
            fd_tol = 1e-10 if precision == "64" else 2e-4
            assert_allclose(energies[:, 1:], energies[:, :1], tol=tol)
            assert_allclose(gradients[:, 1:], gradients[:, :1], tol=tol)
            assert_allclose(tangent[:n_cases], tangent[:n_cases].transpose(0, 2, 1), tol=tol)
            ad_gradient_error = 0.0
            ad_hessian_error = 0.0
            for i_e in range(n_cases):
                F = torch.tensor(deformations[i_e], dtype=gs.tc_float, requires_grad=True)
                mu, lam = coefficients[i_e]

                def energy_reference(value):
                    # Row triple product keeps autodiff well-defined even at rank-deficient matrices.
                    J = torch.dot(value[0], torch.linalg.cross(value[1], value[2]))
                    return 0.5 * (mu * ((value * value).sum() - 3) + (lam + mu) * (J - 1 - mu / (lam + mu)) ** 2)

                gradient = torch.autograd.functional.jacobian(energy_reference, F).numpy()
                reference = torch.autograd.functional.hessian(energy_reference, F).reshape(9, 9).numpy()
                ad_gradient_error = max(ad_gradient_error, np.abs(gradients[i_e, 0] - gradient).max() / scale[i_e])
                ad_hessian_error = max(ad_hessian_error, np.abs(tangent[i_e] - reference).max() / scale[i_e])
                assert_allclose(gradients[i_e, 0] / scale[i_e], gradient / scale[i_e], tol=tol)
                assert_allclose(tangent[i_e] / scale[i_e], reference / scale[i_e], tol=tol)
            fd_gradient = (
                ((energies[n_cases : 10 * n_cases, 0] - energies[10 * n_cases :, 0]) / (2 * eps)).reshape(9, n_cases).T
            )
            fd_hessian = (
                ((gradients[n_cases : 10 * n_cases, 0] - gradients[10 * n_cases :, 0]) / (2 * eps))
                .reshape(9, n_cases, 9)
                .transpose(1, 2, 0)
            )
            gradient_error = np.abs(fd_gradient - gradients[:n_cases, 0].reshape(n_cases, 9)) / scale[:, None]
            hessian_error = np.abs(fd_hessian - tangent[:n_cases]) / scale[:, None, None]
            assert_allclose(gradient_error, 0, atol=fd_tol)
            assert_allclose(hessian_error, 0, atol=fd_tol)
            objectivity_error = 0.0
            for start in range(0, n_cases, 16):
                assert_allclose(energies[start : start + 8], energies[start + 8 : start + 16], tol=tol)
                rotated = rotation @ gradients[start : start + 8, 0] / scale[start]
                observed = gradients[start + 8 : start + 16, 0] / scale[start]
                objectivity_error = max(objectivity_error, np.abs(rotated - observed).max())
                assert_allclose(rotated, observed, tol=tol)
                assert_allclose(gradients[start, 0], 0, atol=tol * scale[start])
                mu, lam = coefficients[start]
                # Include volumetric and shear strain so both physical Lame parameters are checked.
                delta = np.array([[0.2, 0.3, 0], [0.1, -0.1, 0.2], [0, -0.2, 0.4]], dtype=gs.np_float)
                expected = mu * (delta + delta.T) + lam * np.trace(delta) * np.eye(3)
                assert_allclose((tangent[start] @ delta.reshape(9)).reshape(3, 3), expected, tol=tol)
            print(
                f"{backend}/{precision}: {n_cases} states; normalized AD errors "
                f"gradient={ad_gradient_error:.3g}, Hessian={ad_hessian_error:.3g}; "
                f"FD errors gradient={gradient_error.max():.3g}, Hessian={hessian_error.max():.3g}; "
                f"objectivity={objectivity_error:.3g}"
            )
        finally:
            if owns_runtime:
                gs.destroy()


if __name__ == "__main__":
    unittest.main()
