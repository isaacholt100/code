import math
from scipy.spatial import Delaunay
import numpy as np

from custom_types import FloatArray

def _f(x: float) -> float:
    if x > 0:
        return math.exp(-1/x)
    return 0.0

def _g(x: float) -> float:
    return _f(x) / (_f(x) + _f(1 - x))

def _h(x: float) -> float:
    return 1/(1 + np.exp(-15*(x - 0.5))) # use np.exp as math.exp throws range error for large input values

def smooth_transition(a: float, b: float, x: float) -> float: # smooth transition from 0 to 1 on the interval [a, b]
    assert a <= b
    if a == b and x <= a:
        return 0
    if a == b and x >= b:
        return 1
    return _g((x - a)/(b - a))

def single_matched_cusp_approx_weight(delta_p: float, delta_q: float, cusp: FloatArray, x: FloatArray) -> float:
    assert cusp.ndim == 1
    assert x.ndim == 1
    assert cusp.shape == x.shape
    assert delta_p <= delta_q
    # assert len(cusp) == len(x)
    norm = float(np.linalg.norm(x - cusp))
    return smooth_transition(delta_p, delta_q, norm)

def matched_cusps_approx_weight(delta_p: float, delta_q: float, matched_cusp_points: list[FloatArray], x: FloatArray) -> float:
    assert x.ndim == 1
    assert all(cusp.ndim == 1 for cusp in matched_cusp_points)

    assert delta_p <= delta_q
    return math.prod(single_matched_cusp_approx_weight(delta_p, delta_q, cusp, x) for cusp in matched_cusp_points)

def _barycentric_coords(triangulation: Delaunay, simplex_index: int, x: FloatArray) -> list[float]:
    assert x.ndim == 1

    b = triangulation.transform[simplex_index, :2].dot(np.transpose(x - triangulation.transform[simplex_index, 2]))
    b = b.tolist()
    b.append(1 - b.sum(axis=0))
    return b

def simplex_approx_weight(
    delta_p: float,
    delta_q: float,
    triangulation_points: FloatArray,
    triangulation: Delaunay,
    x: FloatArray,
    simplex_index: int
) -> float:
    assert triangulation_points.ndim == 2

    assert delta_p <= delta_q
    simplex_vertices = [triangulation_points[i] for i in triangulation.simplices[simplex_index]]
    return math.prod(smooth_transition(0, 0.1, l) for l in _barycentric_coords(triangulation, simplex_index, x)) * math.prod(single_matched_cusp_approx_weight(delta_p, delta_q, vertex, x) for vertex in simplex_vertices)