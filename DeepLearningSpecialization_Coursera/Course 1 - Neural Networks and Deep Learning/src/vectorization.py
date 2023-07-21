import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc1 = 1000*(time.time()-tic)

print("Vectorized version:\t", str(toc1), "ms")

c = 0
tic = time.time()
for i in range(len(a)):
    c += a[i] * b[i]
toc2 = 1000*(time.time()-tic)

print("For loop:\t\t", str(toc2), "ms")

print("Vec is",str(toc2/toc1),"times faster than for")


for i in range(10):
    print(i, ' ')
