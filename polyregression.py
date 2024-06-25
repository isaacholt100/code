import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x) + np.sinh(x) + np.abs(x)

linspace = np.linspace(-3, 3, 5000)
sample_points = np.linspace(-3, 3, 10000)
regression_degree = 6
data = [(x, f(x)) for x in sample_points]
sample_values = [d[1] for d in data]
design_matrix = np.matrix([[x**i for i in range(regression_degree + 1)] for x in sample_points])
estimated_polynomial_regression_coefficients = (np.linalg.inv(np.transpose(design_matrix) @ design_matrix) @ np.transpose(design_matrix) @ sample_values).flatten().tolist()[0]

def polynomial_estimate(x):
    return sum(estimated_polynomial_regression_coefficients[i] * x**i for i in range(len(estimated_polynomial_regression_coefficients)))

plt.plot(linspace, f(linspace), label="f(x)", color="blue")
plt.plot(linspace, [polynomial_estimate(x) for x in linspace], label="estimate(x)", color="red")
plt.show()