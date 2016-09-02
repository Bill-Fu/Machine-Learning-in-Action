#coding:utf-8

from numpy import *
from os import listdir
import operator

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0] #shape是numpy里array数组类型的，返回数组的行数和列数
    
    #距离计算
    diffMat=tile(inX,(dataSetSize,1))-dataSet #numpy中函数tile(A,n)是重复数组A共n次
    sqDiffMat=diffMat**2 #每个元素都乘方
    sqDistances=sqDiffMat.sum(axis=1) #numpy函数sum求和函数，axis=0是按列求和，axis=1是按行求和
    distances=sqDistances**0.5 #每个元素开根号

    sortedDistIndicies=distances.argsort() #argsort是从小到大排的索引值
    classCount={} #创建一个空的字典
    
    #选择距离最小的k个点
    for i in range(k): 
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #字典 dict.get(key,defalut=None)返回一个给定key对应的键值，如果key不存在，则返回默认值，这里表示返回0

    #排序
    sortedClassCount=sorted(classCount.iteritems(),
                            key=operator.itemgetter(1),reverse=True) #深究：iteritor是迭代器的意思，一次反悔一个数据项，知道没有为止；itemgetter是获得哪些维度的信息
                                                                     #相当于定义了一个函数，reverse表示升序还是降序排列，默认false是升序排列

    return sortedClassCount[0][0]

def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()     #一行一行的读取
    numberOfLines=len(arrayOLines) #得到文件的行数

    returnMat=zeros((numberOfLines,3)) #创建返回的NumPy矩阵
    classLabelVector=[]
    index=0

    for line in arrayOLines:
        line=line.strip()         #去掉line中某种字符，无参数则为去掉包括换行符，空格等
        listFromLine=line.split('\t')       
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals=dataSet.min(0) #参数0使得从每列找最小值
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals  
    normDataSet=zeros(shape(dataSet)) 
    m=dataSet.shape[0]                  #dataSet行数
    normDataSet=dataSet-tile(minVals,(m,1)) #tile函数把minVals复制成m行，1列
    normDataSet=normDataSet/tile(ranges,(m,1)) #这是指具体特征值相除，在numpy中不表示矩阵除法
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio=0.10 #用于测试的数据比例
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],5)
        print "the classifier came back with:%d,the real answer is:%d"%(classifierResult,datingLabels[i])
        if(classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print "the total error rate is:%f"%(errorCount/float(numTestVecs))

def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(raw_input("percentage of time spent playing video games?"))
    ffMiles=float(raw_input("frequent fliter miles earned per year?"))
    iceCream=float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person:",resultList[classifierResult-1]
    
def img2vector(filename):
    returnVect=zeros((1,1024))     #把一个32*32的图像转换成一个1*1024的向量
    fr=open(filename)          
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')         #获取目录内容
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s' % fileNameStr)
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s' % fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with: %d, the real answer is: %d, the filename is: %s" % (classifierResult,classNumStr,fileNameStr)
        if(classifierResult!=classNumStr):errorCount+=1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
