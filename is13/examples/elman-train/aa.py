import numpy
import math,time
s=time.time()
a=numpy.load('aa/W.txt.npy')
print a.shape  #a shape:(100L,12L)
# print numpy.mean(a,axis=0)
# print numpy.linalg.norm([12,2,5,7])

b=numpy.array([4,5,6])
b2=numpy.array([1,2,3])


print math.exp(-numpy.linalg.norm(b2-b)/12),math.e

print 'costs:',(time.time()-s)/60
t=(1,23)
print t[1]