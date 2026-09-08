# SPDX-License-Identifier: Apache-2.0
"""Measure complete reference and device logit rows.

Inputs: equal, finite rows and fixed calibration bounds.
Outputs: float64 errors, token margins and acceptance results.
Errors: ValueError for invalid rows, bounds or calibration inputs.
"""
import math

import numpy as np


METRICS = ("linf", "relative_l2", "total_variation")
FACTOR = 1.5


def measure(reference, device):
    reference = np.asarray(reference, dtype=np.float64)
    device = np.asarray(device, dtype=np.float64)
    if (reference.ndim != 1 or device.shape != reference.shape or reference.size < 2
            or not np.isfinite(reference).all() or not np.isfinite(device).all()):
        raise ValueError("invalid or non-finite logit row")
    norm = float(np.linalg.norm(reference))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError("invalid reference norm")
    error = device - reference
    rp = np.exp(reference - reference.max())
    dp = np.exp(device - device.max())
    rp /= rp.sum()
    dp /= dp.sum()
    top = np.partition(reference, -2)[-2:]
    result = {
        "linf": float(np.abs(error).max()),
        "relative_l2": float(np.linalg.norm(error) / norm),
        "total_variation": float(np.abs(rp - dp).sum() * 0.5),
        "reference_top": int(reference.argmax()),
        "device_top": int(device.argmax()),
        "reference_gap": float(top[1] - top[0]),
    }
    if any(not math.isfinite(result[key]) for key in METRICS):
        raise ValueError("non-finite error measure")
    return result


def validate_bounds(bounds):
    if set(bounds) != set(METRICS):
        raise ValueError("incomplete error bounds")
    if any(type(bounds[key]) not in (float, int) or not math.isfinite(bounds[key])
           or bounds[key] < 0 for key in METRICS):
        raise ValueError("invalid error bound")
    if bounds["total_variation"] >= 1:
        raise ValueError("the probability bound must be below one")


def calibrate(rows):
    maxima = {key: 0.0 for key in METRICS}
    count = 0
    for row in rows:
        if row["split"] != "calibration":
            raise ValueError("a validation row cannot set a bound")
        for key in METRICS:
            value = row[key]
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid calibration measure")
            maxima[key] = max(maxima[key], value)
        count += 1
    if count == 0:
        raise ValueError("no calibration rows")
    bounds = {key: FACTOR * value for key, value in maxima.items()}
    validate_bounds(bounds)
    return {"rows": count, "factor": FACTOR, "maxima": maxima, "bounds": bounds}


def assess(row, bounds):
    validate_bounds(bounds)
    if any(not math.isfinite(row[key]) or row[key] < 0 for key in METRICS):
        raise ValueError("invalid error measure")
    clear = row["reference_gap"] > 2.0 * bounds["linf"]
    failed = [key for key in METRICS if row[key] > bounds[key]]
    if clear and row["reference_top"] != row["device_top"]:
        failed.append("clear_argmax")
    return {"clear": clear, "failed": failed}
