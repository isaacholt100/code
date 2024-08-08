
from cusps.find_cusps import cluster_and_estimate
from cusps.reconstruct import hypercuboid_vertices
from polyapprox.approx import max_degree_indices


def test_cluster_and_estimate():
    assert cluster_and_estimate([(0, 0.1), (1, 0.5), (0.1, 0.2), (-0.1, 0.05), (1.1, 0.4), (2, 0.2)], 0.11) == [-0.1, 1.1, 2]

def test_max_degree_indices():
    assert sorted(max_degree_indices(3, 2)) == sorted([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [3, 0]])

