import math
from scipy.spatial import Delaunay
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

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
    return _h((x - a)/(b - a))

def single_matched_cusp_approx_weight(delta_p: float, delta_q: float, cusp: FloatArray, x: FloatArray) -> float:
    assert delta_p <= delta_q
    # assert len(cusp) == len(x)
    norm = float(np.linalg.norm(x - cusp))
    if norm >= delta_q:
        return 1.0
    if norm <= delta_p:
        return 0.0
    return smooth_transition(delta_p, delta_q, norm)

def matched_cusps_approx_weight(delta_p: float, delta_q: float, matched_cusp_points: list[FloatArray], x: FloatArray) -> float:
    assert delta_p <= delta_q
    return math.prod(single_matched_cusp_approx_weight(delta_p, delta_q, cusp, x) for cusp in matched_cusp_points)

def _barycentric_coords(x, x1, x2, x3):
    det = (x2[1] - x3[1])*(x1[0] - x3[0]) + (x3[0] - x2[0])*(x1[1] - x3[1])
    lambda1 = ((x2[1] - x3[1])*(x[0] - x3[0]) + (x3[0] - x2[0])*(x[1] - x3[1])) / det
    lambda2 = ((x3[1] - x1[1])*(x[0] - x3[0]) + (x1[0] - x3[0])*(x[1] - x3[1])) / det
    lambda3 = 1 - lambda1 - lambda2
    return [lambda1, lambda2, lambda3]

def _barycentric_coords_new(triangulation: Delaunay, simplex_index: int, x):
    b = triangulation.transform[simplex_index, :2].dot(np.transpose(x - triangulation.transform[simplex_index, 2]))
    return b.tolist() + [1 - b.sum(axis=0)]

def simplex_approx_weight(delta_p: float, delta_q: float, triangulation_points: list, triangulation: Delaunay, x, simplex_index: int) -> float:
    assert delta_p <= delta_q
    simplex_vertices = [triangulation_points[i] for i in triangulation.simplices[simplex_index]]
    return math.prod(smooth_transition(-0.01, 0, l) for l in _barycentric_coords_new(triangulation, simplex_index, x)) * math.prod(single_matched_cusp_approx_weight(delta_p, delta_q, vertex, x) for vertex in simplex_vertices)