import numpy
import io
import os
import mlpy
import random

def classify(data):
    data=numpy.array(data)-numpy.array(mu)
    x=numpy.dot(data,pca_matrix)
    k=numpy.array([x[0]*x[1],x[1]*x[2],x[0]*x[2],x[3]*x[0],x[3]*x[1],x[3]*x[2]])
    x=numpy.append(x,k)
    x=numpy.append([1],x)
    all_theta=numpy.array([numpy.array(paper_theta), numpy.array(rock_theta),numpy.array(scissors_theta)])
    pred=numpy.dot(x,all_theta.T)
    index=numpy.argmax(pred)
    print('index = ',index)
    return index
def learn(data, y):
    y=numpy.array(y,dtype=numpy.int)-1
    #y=1
    mu=numpy.mean(data,0)
    data = data-numpy.tile(mu,(numpy.size(data,0),1))
    data=data/numpy.tile(numpy.std(data,0),(numpy.size(data,0),1))
    randperm=random.sample(range(0,numpy.size(data,0)),numpy.size(data,0))
    data=data[randperm,:]
    y=y[randperm,:]
    pca=mlpy.PCA()
    pca.learn(data)
    data_pca=pca.transform(data,k=5)
    X=data_pca
    idx=mlpy.cv_kfold(numpy.size(X,0), numpy.size(X,0), strat=None, seed=4)
    corrects=0
    falses=0
    confusion=numpy.zeros((3,3))
    for tr,ts in idx:
        X_tr=data_pca[tr,:]
        Y_tr=y[tr]
        X_ts=data_pca[ts,:]
        Y_ts=y[ts]
        
        logReg=mlpy.LibLinear(solver_type='mcsvm_cs')
        logReg.learn(X_tr,Y_tr)
        w=logReg.w()
        y_training_pred=logReg.pred(X_tr)
        y_trained=y_training_pred.copy()
        
        print(numpy.size(numpy.nonzero(y_trained==Y_tr))/float(numpy.size(Y_tr)))
        y_dat=logReg.pred(X_ts)
        


        corrects=corrects+numpy.size(numpy.nonzero(y_dat==Y_ts))
        falses=falses+numpy.size(Y_ts)-numpy.size(numpy.nonzero(y_dat==Y_ts))
        confusion[y_dat,Y_ts]=confusion[y_dat,Y_ts]+1

    print(falses)
    print(corrects)
    print(corrects/float(falses+corrects))
    print(confusion)


def script():
    [X,y]=read_file()
    print(X)
    print(y)
    learn(X,y)
def read_file():
    read=file('output.txt','r')
    full_data=None
    while(1):
        data=read.readline()
        if (data==''):
            break
        arr=data.split(',')
        set=numpy.array([])
        for i in arr:
            if(i=='\r\n'):
                continue
            set=numpy.append(set,float(i))
        
        if(full_data!=None):
            full_data=numpy.vstack((full_data,set))
        else:
            full_data=set
    print(numpy.size(full_data,0))
    return (full_data[:,0:-1],full_data[:,-1])
#script()



