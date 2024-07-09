def gap_weighted_error(surface, approximation):
    def delta(s, i: int, j: int):
        return s[i] - s[j]
    epsilon_w = 5e-2
    max_error = 0
    for i in range(len(approximation)):
        for j in range(len(approximation)):
            if i == j:
                continue
            m = max(abs(delta(approximation, i, j) - delta(surface, i, j))/(epsilon_w + abs(delta(surface, i, j))))
            if m > max_error:
                max_error = m
    return max_error

def max_abs_errors(original, reconstructed):
    return [max(abs(reconstructed[j, i] - original[j, i]) for i in range(reconstructed.shape[1])) for j in range(len(original) - 1)]

def sum_max_abs_error(original, reconstructed):
    return sum(max_abs_errors(original, reconstructed))