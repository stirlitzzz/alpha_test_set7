# import pstats

# p = pstats.Stats("stats.txt")
# p.strip_dirs().sort_stats("cumulative").print_stats(50)

import numpy as np

arr1 = np.array([1,0,np.nan,np.inf,-np.inf])
print(arr1/1) #[  1.   0.  nan  inf -inf]
print(arr1/0) #[ inf  nan  nan  inf -inf]
print(arr1/np.nan) #[nan nan nan nan nan]
print(arr1/np.inf) #[ 0.  0. nan nan nan]
print(arr1/-np.inf) #[-0. -0. nan nan nan]

print(np.nan_to_num(arr1,nan=0,posinf=0,neginf=0)) #[1. 0. 0. 0. 0.]
