from cmath import sin
import numpy as np

## To  get implement an array
a = np.array([1,21,56], dtype='int16')
print(a)

b = np.array([[1,2,3], [1.3,4.5,4.0], [3,4,5]]);
print(b)

## To get dimension
print(a.ndim)
print(b.ndim)

## to the the order of matrice

print(a.shape)
print(b.shape)

## to get type of matrice 

print(a.dtype)
print(b.dtype)

## size of the matrice

print(a.itemsize)
print(b.itemsize)

## total size ie the number of element

print(a.size)
print(b.size)

## bytes * element ie sum(singleElement * its bytes)

print(a.nbytes)
print(b.nbytes)

## accesing elemnets

a = np.array([[1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16,17,18]])

## index starts from 0 and not one , it's a programming language

print(a)
print(a.shape)
print(a[:,:])         ## whole matrice
print(a[1,2])         ## first row second column
print(a[:,0])         ## first coloumn
print(a[0, :])        ## first row
print(a[1, 5:8:2])    ## startIndex:endIndex:stepIndex

## 3D eg

A = np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]], [[13,14],[15,16],[17,18]]]);

print(A)
print(A.shape)
print(A[1,2,:])

## MATRICES OF DIFFERENT TYPES

B = np.zeros((2,2))
print(B)

C = np.ones((3,3))
print(C)

D = np.identity(5)
print(D)

## Matrices of your own Choice

E = np.full((2,3), 56)
print(E)

## copy matrice order/shape

F = np.full(a.shape, 69)
print(F)

F = np.full_like(a, 69)
print(F)

## To create a matrice with random elements
## By default returns a float matrice

G = np.random.rand(40,40)
print(G)

G = np.random.random_sample(a.shape)
print(G)

G = np.random.random_integers(54,69, a.shape)
print(G)

G = np.random.randint(6,40,(40,4))
print(G)

## Repeating array

H = np.repeat(a,5, axis=0)
print(H)


# np.linalg.pinv  --- puedo inv

print('Puedo Inv\n', np.linalg.pinv(a))

# To print a random matrice with some patttern

output = np.ones((5,5))
inside = np.zeros((3,3))
inside[1][1] = 9
output[1:4, 1:4] = inside

print(output)

# Be carefull while copying

a = np.array([1,2,3])
b = a
print('a', a)
print('b', b)

b[0] = 100

print("after changing b", b)
print("not changing a", a)

# use a.copy() to create duplicate

c = a.copy()

c[1] = 10
print("After changing c", c)
print("After changing c, a", a)

# Using math lib of numpy

z = np.array([1,2,3])
v = np.sin(z)
print(v)

# np.dot(a,b)  -> to do matrice vector and matrice matrice multiplication
# np.matmul(a,b)  -> to do matrice vector and matrice matrice multiplication

a = np.ones((2,3))
b = np.full((3,2), 10)

print(np.matmul(a,b))
print(np.dot(a,b))

# det
a = np.array([[1,0,0], [0,32,0], [0,0,65]])
print(np.linalg.det(a))


# eigen values and vectors

la, e = np.linalg.eig(a)
print(la)
print(e)

# inverse

print(np.linalg.inv(a))
print(np.linalg.pinv(a))      # psuedo inverse

# matrice reshape

print(a.reshape((9,1)))

#  stacking vectors

a = np.array([1,2,3])
b = np.array([1,2,3])
c = np.array([1,2,3])

# vertical stacking

print(np.vstack((a,b,c)))

# horizontal stacking

print(np.hstack((a,a,b,b)))

#  generate data from file

# fileData = np.genfromtxt('', dtype='int32', delimiter=',')
# np.genfromtxt()
# print(fileData)