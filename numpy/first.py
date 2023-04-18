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
