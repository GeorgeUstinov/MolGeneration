from __future__ import annotations

import math
from statistics import fmean
from typing import Iterable, Sequence


def trapezoid_auc(wavelengths: Sequence[float], absorbance: Sequence[float], low: float, high: float) -> float:
    if len(wavelengths) != len(absorbance) or len(wavelengths) < 2:
        raise ValueError("wavelength and absorbance arrays must have equal length >= 2")
    if any(b <= a for a, b in zip(wavelengths, wavelengths[1:])):
        raise ValueError("wavelengths must be strictly increasing")
    if low >= high:
        raise ValueError("integration interval must have positive width")
    points: list[tuple[float, float]] = []
    for x, y in zip(wavelengths, absorbance):
        if low <= x <= high:
            points.append((float(x), float(y)))
    for edge in (low, high):
        if not any(abs(x - edge) < 1e-12 for x, _ in points):
            for index in range(len(wavelengths) - 1):
                left, right = wavelengths[index], wavelengths[index + 1]
                if left <= edge <= right:
                    ratio = (edge - left) / (right - left)
                    y = absorbance[index] + ratio * (absorbance[index + 1] - absorbance[index])
                    points.append((edge, float(y)))
                    break
    points.sort()
    return sum((x2 - x1) * (y1 + y2) / 2.0 for (x1, y1), (x2, y2) in zip(points, points[1:]))


def critical_wavelength(wavelengths: Sequence[float], absorbance: Sequence[float], fraction: float = 0.90) -> float:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    total = trapezoid_auc(wavelengths, absorbance, wavelengths[0], wavelengths[-1])
    if total <= 0.0:
        return float(wavelengths[0])
    target = total * fraction
    cumulative = 0.0
    for (x1, y1), (x2, y2) in zip(zip(wavelengths, absorbance), zip(wavelengths[1:], absorbance[1:])):
        segment = (x2 - x1) * (y1 + y2) / 2.0
        if cumulative + segment >= target:
            if segment <= 0.0:
                return float(x2)
            return float(x1 + (x2 - x1) * (target - cumulative) / segment)
        cumulative += segment
    return float(wavelengths[-1])


def beer_lambert_transmittance(absorbance: Iterable[float], loading_scale: float = 1.0) -> float:
    values = [10.0 ** (-max(0.0, float(value)) * loading_scale) for value in absorbance]
    return fmean(values) if values else 1.0


def sigmoid(value: float, midpoint: float, scale: float) -> float:
    if scale <= 0:
        raise ValueError("scale must be positive")
    exponent = max(-60.0, min(60.0, -(value - midpoint) / scale))
    return 1.0 / (1.0 + math.exp(exponent))


def interval_score(value: float, low: float, high: float, softness: float) -> float:
    if not low < high:
        raise ValueError("low must be less than high")
    return sigmoid(value, low, softness) * sigmoid(high - value, 0.0, softness)


def weighted_geometric_mean(
    components: dict[str, float],
    weights: dict[str, float],
    floor: float = 1e-3,
) -> float:
    if not components:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for key, raw in components.items():
        weight = float(weights.get(key, 1.0))
        if weight <= 0.0:
            continue
        value = max(floor, min(1.0, float(raw)))
        numerator += weight * math.log(value)
        denominator += weight
    return math.exp(numerator / denominator) if denominator else 0.0


def lower_confidence_bound(mean: float, std: float, z: float = 1.645) -> float:
    return float(mean) - float(z) * max(0.0, float(std))


def upper_confidence_bound(mean: float, std: float, z: float = 1.645) -> float:
    return float(mean) + float(z) * max(0.0, float(std))

