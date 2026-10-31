"""Evaluate a mesh study independently of time integration or contact acceptance.

Input is a JSON manifest of at least three mesh levels and adjacent-level errors.
Errors must be measured on a common physical domain, load path and probe set.
This is a validation utility; it neither changes the FEM solver nor certifies CCD.
"""

import argparse
import itertools
import json
import math
from pathlib import Path


def finite_number(value, minimum=0):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= minimum


def assess(study, relative_tolerance=0.02, displacement_atol=5e-6, force_residual_tolerance=1e-3):
    """Conservative spatial-convergence gate; thresholds are engineering policy."""
    if not all(
        finite_number(value) and value > 0
        for value in (relative_tolerance, displacement_atol, force_residual_tolerance)
    ):
        raise ValueError("Positive finite acceptance tolerances required")
    reasons = []
    levels = study.get("levels", [])
    pairs = study.get("pairs", [])
    if len(levels) < 3 or len(pairs) != len(levels) - 1:
        reasons.append("insufficient_levels_or_pairs")
    for flag in ("common_domain_verified", "common_support_verified", "identical_configuration", "load_path_matched"):
        if study.get(flag) is not True:
            reasons.append(flag + "_missing")
    for level in levels:
        name = str(level.get("id", "unknown"))
        if level.get("complete") is not True:
            reasons.append(name + ":integration_incomplete")
        if level.get("sap_converged") is not True or level.get("fem_prediction_converged") is not True:
            reasons.append(name + ":subproblem_not_converged")
        jacobian = level.get("min_jacobian")
        if not finite_number(jacobian) or jacobian <= 0:
            reasons.append(name + ":invalid_or_inverted_elements")
        residual = level.get("max_relative_force_residual")
        if not finite_number(residual) or residual > force_residual_tolerance:
            reasons.append(name + ":full_step_force_residual")
        for metric in ("nodes", "elements", "h_rms_m", "h_max_m"):
            if not finite_number(level.get(metric)) or level[metric] <= 0:
                reasons.append(name + ":invalid_" + metric)
    for coarse, fine in itertools.pairwise(levels):
        if (
            not all(
                finite_number(level.get(key)) for level in (coarse, fine) for key in ("elements", "h_rms_m", "h_max_m")
            )
            or fine["elements"] <= coarse["elements"]
            or fine["h_rms_m"] >= coarse["h_rms_m"]
            or fine["h_max_m"] >= coarse["h_max_m"]
        ):
            reasons.append("no_spatial_refinement")
    metrics = ("displacement_relative_l2", "displacement_max_m", "reaction_relative", "energy_relative")
    for pair in pairs:
        if any(not finite_number(pair.get(metric)) for metric in metrics):
            reasons.append("invalid_or_missing_pair_metrics")
    if pairs:
        finest = pairs[-1]
        for metric in metrics:
            value = finest.get(metric)
            threshold = displacement_atol if metric == "displacement_max_m" else relative_tolerance
            if not finite_number(value) or value > threshold:
                reasons.append("finest_" + metric)
        for previous, current in itertools.pairwise(pairs):
            for metric in metrics:
                older, newer = previous.get(metric), current.get(metric)
                if not finite_number(older) or not finite_number(newer) or (newer >= older and newer > 0):
                    reasons.append("nondecreasing_" + metric)
    return {
        "mesh_convergence_accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "policy": {
            "relative_tolerance": relative_tolerance,
            "displacement_atol_m": displacement_atol,
            "force_residual_tolerance": force_residual_tolerance,
        },
        "strict_nonpenetration_accepted": False,
        "contact_note": "This spatial gate provides no continuous-collision or nonintersection evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    parser.add_argument("--displacement-atol", type=float, default=5e-6)
    parser.add_argument("--force-residual-tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    result = assess(
        json.loads(args.input.read_text()),
        args.relative_tolerance,
        args.displacement_atol,
        args.force_residual_tolerance,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))
    if not result["mesh_convergence_accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
