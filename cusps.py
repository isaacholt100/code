import matplotlib.pyplot as plt
import math
import numpy as np
from math import sin, cos

def value_sorted(x):
    return sorted([2/3*sin(x), sin(2*x), cos(2*x), 1 + cos(4*x/3)])

# x_plot = np.linspace(0, 2, 10000)
# a = 1.438
# c = 2*(sin(a)*cos(a) - (1 + cos(4/3*a)) * -4/3*sin(4/3 * a))/(-4/3*sin(4/3*a) - cos(a))
# y_plot = [[y**2 + c*y for y in value_sorted(x)] for x in x_plot] #[np.pow(np.add(value_sorted(x), 3) / 4, 32) for x in x_plot]
# plt.plot(x_plot, [y[0] for y in y_plot])
# plt.plot(x_plot, [y[1] for y in y_plot])
# plt.plot(x_plot, [y[2] for y in y_plot])
# # plt.plot(x_plot, [y[3] for y in y_plot])
# plt.show()

# IDEA: by setting p(x) = x^2 + cx where c = 2 (h(a) h'(a) - g(a) g'(a))/(g'(a) - h'(a)), a is the point where the function f is not differentiable, we have that p(f(x)) is differentiable and we can recover x (assuming x is positive) by x = -c/2 + sqrt(c^2 / 4 + y).

epsilon = 0.05
nodes = np.array([2, 4, 10])
e = 5
start = 1
end = 12
def s(x):
    return 2 - 2*x**4



delta = 2 * epsilon
b = nodes - epsilon
c = nodes + epsilon
a = b - delta
d = c + delta

p_2 = (e + 1)/(2*delta)
p_1 = 1 - 2 * p_2 * b
p_0 = p_2 * b**2
def p(i: int):
    return lambda x: p_2 * x**2 + p_1[i] * x + p_0[i]
p_a = (e + 1)/2 * delta + a # p_i (a_i)

q_2 = (e - 1)/(2 * delta)
q_1 = 1 - 2 * q_2 * c
q_0 = q_2 * c**2
def q(i: int):
    return lambda x: q_2 * x**2 + q_1[i] * x + q_0[i]
q_d = (e - 1)/2 * delta + d # q_i (d_i)

n = 10
m = np.array([start, *[(d[i] + a[i + 1])/2 for i in range(len(nodes) - 1)], end])
K = 5
k = m + K
s_0 = s(0)
s_1 = 8 # s_1 = s'(-1)
abg_v = [np.linalg.inv([[0, 1, -1], [s_0, 1, 0], [s_1 / (m[i + 1] - d[i]), 0, (2*n)/(m[i + 1] - d[i])]]) @ [q_d[i], k[i + 1], e] for i in range(len(nodes))] # NOTE: m, k indexed by i + 1 as indices start at 0
alpha = [t[0] for t in abg_v]
beta = [t[1] for t in abg_v]
gamma = [t[2] for t in abg_v]

def v(i: int):
    return lambda x: alpha[i] * s((x - d[i])/(m[i + 1] - d[i]) - 1) + beta[i] - gamma[i] * ((x - d[i])/(m[i + 1] - d[i]) - 1)**(2*n) # NOTE: m indexed by i + 1 as indices start at 0

ABC_u_rev = [np.linalg.inv([[0, 1, -1], [s_0, 1, 0], [s_1 / (a[i] - m[i]), 0, (2*n)/(a[i] - m[i])]]) @ [p_a[i], k[i], e] for i in range(len(nodes))] # NOTE: m indexed by i as indices start at 0

A = [t[0] for t in ABC_u_rev]
B = [t[1] for t in ABC_u_rev]
C = [t[2] for t in ABC_u_rev]

def u_rev(i: int):
    return lambda x: A[i] * s((x - -a[i])/(-m[i] - -a[i]) - 1) + B[i] - C[i] * ((x - -a[i])/(-m[i] - -a[i]) - 1)**(2*n)

def u(i: int):
    return lambda x: u_rev(i)(-x)

def f(x):
    for i in range(len(nodes)):
        if m[i] <= x and x < a[i]: # NOTE: m indexed by i as indices start at 0
            return u(i)(x)
        if a[i] <= x and x < b[i]:
            return p(i)(x)
        if b[i] <= x and x <= c[i]:
            return x
        if c[i] < x and x <= d[i]:
            return q(i)(x)
        if d[i] < x and x <= m[i + 1]: # NOTE: m indexed by i + 1 as indices start at 0
            return v(i)(x)
    return -1

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_aspect("equal")

id_plot = np.linspace(start, end, 10000)
ax.plot(id_plot, id_plot, color="gray", linestyle=":", label="x")

for i in range(len(nodes)):
    N = 10000
    x_plot = np.linspace(m[i], a[i], N)
    ax.plot(x_plot, [u(i)(x) for x in x_plot], color="blue", label="u_i (x)")
    x_plot = np.linspace(a[i], b[i], N)
    ax.plot(x_plot, [p(i)(x) for x in x_plot], color="red", label="p_i (x)")
    x_plot = np.linspace(b[i], c[i], N)
    ax.plot(x_plot, x_plot, color="green", label="x")
    x_plot = np.linspace(c[i], d[i], N)
    ax.plot(x_plot, [q(i)(x) for x in x_plot], color="purple", label="q_i (x)")
    x_plot = np.linspace(d[i], m[i + 1], N)
    ax.plot(x_plot, [v(i)(x) for x in x_plot], color="orange", label="v_i (x)")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig("method1.png", dpi=400)
plt.show()

def method2():
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.set_aspect("equal")
    epsilon1 = 0.2
    b = [*(nodes - epsilon1), end]
    c = [start, *(nodes + epsilon1)]

    D = 10
    n = 20
    def t(x: float):
        return D * (1 - math.sin(math.pi * x)**(2*n))

    def r(i: int):
        return lambda x: t((x - c[i])/(b[i] - c[i]) + 1/2) + x

    for i in range(len(nodes)):
        N = 10000
        x_plot = np.linspace(c[i], b[i], N) # NOTE: different indexing as starts at 0
        ax.plot(x_plot, [r(i)(x) for x in x_plot], color="red")
        x_plot = np.linspace(b[i], c[i + 1])
        ax.plot(x_plot, x_plot, color="blue")
    plt.show()

method2()