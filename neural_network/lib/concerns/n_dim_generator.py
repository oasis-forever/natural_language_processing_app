import random as r

def generate_n_dim(dim):
    p_float = []
    n_float = []
    for i in range(0, 11):
        p_float.append(float(i) / 10)
    for i in p_float:
        n_float.append(i * -1)
    p_float.sort(reverse=True)
    n_float.pop(0)
    float_list = p_float + n_float
    n_dim = r.choices(float_list, k=dim)
    return n_dim
