import kNN
from numpy import *
import operator
def classify(newdata, dataset, labels, k):
    dataSize=dataset.shape[0]
    print dataSize
    diffMat=tile(newdata,(dataSize,1)) - dataset
    sqDiffMat=diffMat**2
    sqDistance=sqDiffMat.sum(axis=1)
    distance=sqDistance**0.5
    sortedDist=distance.argsort()
    classCount={}
    for i in range(k):
        votelabel=labels[sortedDist[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1
    sortedClass=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClass[0][0]

group, labels=kNN.createDataset()
print classify([0,0],group,labels,3)
print tile([0,0],5)
print tile([0,0],(1,1))

a = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]


print sorted(a, key=lambda s:s[1])