# How calculate time 

import time

t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1 - t0))

## Function

def time_spent(time0):
    t = time.time() - time0
    t_int = int(t) // 60
    t_min = t % 60
    if(t_int != 0):
        return '{}min {:.3f}s'.format(t_int, t_min)
    else:
        return '{:.3f}s'.format(t_min)

"""
HOW TO USE
t0 = time.time()
print(time_spent(t0))
"""
