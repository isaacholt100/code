import itertools

from custom_types import FloatArray

epsilon_w = 5e-2

def delta(s: FloatArray, i: int, j: int) -> float:
    assert s.ndim == 1

    return s[i] - s[j]

def gap_weighted_error(original: FloatArray, reconstructed: FloatArray) -> float:
    assert original.ndim == 2
    assert reconstructed.ndim == 2

    return max(gap_weighted_error_pointwise(original, reconstructed))

def gap_weighted_error_pointwise(original: FloatArray, reconstructed: FloatArray) -> list[float]:
    assert original.ndim == 2
    assert reconstructed.ndim == 2

    # assert surface.shape == approximation.shape
    errors = []
    for k in range(reconstructed.shape[0]):
        surface_values = original[k, :]
        approx_values = reconstructed[k, :]
        error = max(abs(delta(surface_values, i, j) - delta(approx_values, i, j))/(epsilon_w + abs(delta(surface_values, i, j))) for (i, j) in itertools.product(range(reconstructed.shape[1]), range(reconstructed.shape[1])))
        errors.append(error)
    return errors

def max_abs_errors(original: FloatArray, reconstructed: FloatArray) -> list[float]:
    assert original.ndim == 2
    assert reconstructed.ndim == 2
    
    return [max(abs(reconstructed[i, j] - original[i, j]) for i in range(reconstructed.shape[0])) for j in range(reconstructed.shape[1])]

def sum_max_abs_error(original: FloatArray, reconstructed: FloatArray) -> float:
    assert original.ndim == 2
    assert reconstructed.ndim == 2

    return sum(max_abs_errors(original, reconstructed))