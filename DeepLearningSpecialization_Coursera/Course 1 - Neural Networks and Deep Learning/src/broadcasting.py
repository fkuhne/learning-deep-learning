import numpy as np

#
# BROADCASTING EXAMPLES
#
A = np.array([[56.0, 0.0, 4.4, 68.0 ],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])
print(A)

cal = A.sum(axis=0) # axis=0 means to sum vertically
print(cal)

percentage = 100*A/(cal.reshape(1,4)) # technically the reshape would not be needed, 
                                      # but sometimes it's worth it to make sure the
                                      # matrix operation will have the arguments of correct size
print(percentage)

B = np.array([[1,2,3,4]]).T
print(B+100)

C = np.array([[1,2,3],
              [4,5,6]])
D = np.array([[100,200,300]])
print(C+D)

E = np.array([[100],
              [200]])
print(C+E)


f = np.random.rand(5) # this creates a rank 1 array
print(f)
print(f.shape) # (5,)
print(f.T)
print(f.T.shape) # (5,)

# use always two dimensions, like this:
f = np.random.rand(5,1) # this ensures that a row or column vector is created
print(f)
print(f.shape) # (5,1)
print(f.T)
print(f.T.shape) # (1,5)

# dont use rank 1 arrays!

# to fix rank 1 arrays, reshape can be used: f = f.reshape(5,1)

# it is also possible to call assert to make sure the dimensions are correct:
assert(f.shape == (5,1))

def loss(y, yhat):
    return -(y*np.log(yhat) + (1-y)*(np.log(1-yhat)))

print(loss(1,0.9))

x=np.array([[[1],[2]],[[3],[4]]])
print(x.shape)

a=np.random.randn(3,4) # a.shape=(3,4)
b=np.random.randn(1,4) # b.shape=(1,4)
c=a+b
print(c.shape)

a=np.random.randn(4,3) # a.shape=(4,3)
b=np.random.randn(1,3) # b.shape=(1,3)
c=a*b
print(c.shape)

a = np.array([[2,1],[1,3]])
print((a*a))