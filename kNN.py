from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDateSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
def classify0(inX,dataSet,labels,k):
    dataSetSize =dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    # change itemgetter te item
    sortedClassCourt=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCourt[0][0]
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    a={'largeDoses':1,'smallDoses':2,'didntLike':3}
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(a.get(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
def datingClassTest():
     hoRatio=0.10
     datingDatMat,datingLabels=file2matrix('datingTestSet.txt')
     norMat,ranges,minVals=autoNorm(datingDatMat)
     m=norMat.shape[0]
     numTestVecs=int (m*hoRatio)
     errorCount=0.0
     for i in range(numTestVecs):
         classifierResult=classify0(norMat[i,:],norMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
         print("the classifier came back with: %d, the real answer is: %d"%(classifierResult,datingLabels[i]))
         if(classifierResult!=datingLabels[i]):
             errorCount+=1.0
         print('the total error rate is :%f'%(errorCount/float(numTestVecs)))
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect
testVector=img2vector('trainingDigits/0_13.txt')
print(testVector[0,0:31])
print(testVector[0,32:63])
# def classifyPerson():
#     resultList=['not at all','in small does','in large does']
#     percentTats=float(input("percentage of time spent playing video games?"))
#     ffMiles=float(input("frequent flier miles earned per year?"))
#     iceCream=float(input("liters of ice cream consumed per year?"))
#     datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
#     normMat,ranges,minVals=autoNorm(datingDataMat)
#     print(normMat,ranges,minVals)
#     inArr=array([ffMiles,percentTats,iceCream])
#     print('inArr:',inArr)
#     classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
#     print('You will probably like this people: ',classifierResult)
# if __name__=='__main__':
#     print('dataset-labels')
#     print(createDateSet())
#     group,labels=createDateSet()
#     label=classify0([1,1.3],group,labels,3)
#     print(label)
# datingDatMat,datingLabels=file2matrix('datingTestSet.txt')
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(datingDatMat[:,1],datingDatMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
# plt.show()
# norMat,ranges,minVals=autoNorm(datingDatMat)
# print('norMat:',norMat)
# print('ranges:',ranges)
# print('minVals:',minVals)
# datingClassTest()
# classifyPerson()

