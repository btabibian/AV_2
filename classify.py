import numpy
import io
import os
import mlpy
import random

def preproccessing(data,y):
    y=numpy.array(y,dtype=numpy.int)-1
    mu=numpy.mean(data,0)
    stan=numpy.std(data,0)
    data = data-numpy.tile(mu,(numpy.size(data,0),1))
    data=data/numpy.tile(stan,(numpy.size(data,0),1))
    
    randperm=random.sample(range(0,numpy.size(data,0)),numpy.size(data,0))
    data=data[randperm,:]
    y=y[randperm,:]
    
    pca=mlpy.PCA()
    pca.learn(data)
    data_pca=pca.transform(data,k=5)
    return (data_pca,y,pca,mu,stan)
def train(x,y):
    logReg=mlpy.LibLinear(solver_type='mcsvm_cs')
    logReg.learn(x,y)
    return logReg
def evaluate(y_model,y_true,confusion=None):
    if(confusion!=None):
        confusion[y_model,y_true]=confusion[y_model,y_true]+1
    return numpy.size(numpy.nonzero(y_model==y_true))/float(numpy.size(y_true)),confusion
def pred_new_instance(model,pca_model,mean,stan,x):
    x = x-mean
    x=  x/stan
    pcad=pca_model.transform(x,5)
    y=model.pred(pcad)
    print('label is '+str(y))


def cross_validation(data,y):
    idx=mlpy.cv_kfold(numpy.size(data,0), numpy.size(data,0), strat=None, seed=4)
    corrects=0
    falses=0
    confusion=numpy.zeros((3,3))
    testavg = 0
    testavgcount = 0
    test_res=0
    for tr,ts in idx:
        X_tr=data[tr,:]
        Y_tr=y[tr]
        X_ts=data[ts,:]
        Y_ts=y[ts]
        
        model=train(X_tr,Y_tr)
        y_training_pred=model.pred(X_tr)
        y_trained=y_training_pred.copy()
        y_dat=model.pred(X_ts)    
        
        acc,conf=evaluate(y_trained,Y_tr)
        testavg = testavg + acc
        acc,confusion=evaluate(y_dat,Y_ts,confusion)
        test_res=test_res+acc
        testavgcount = testavgcount + 1
              
    return confusion, testavg/float(testavgcount), test_res/float(testavgcount)
def learn(data, y):
    
    (X,y,pca_model,mean,stan)=preproccessing(data,y)
    model = train(X,y)

    confusion, training_error,test_error=cross_validation(X,y)
    print('average training error: '+str(training_error))
    print('test error: '+str(test_error))
    print('confusion matrix\n'+str(confusion))
    return model,pca_model,mean,stan
def write_features(x,y):
     write=file('features.txt','a');
     p=(y, x)
     for ii in range(0,numpy.size(x,0)):
         i=x[ii,:]
         j=y[ii]
         for k in i:
             write.write(str(k)+',')
         write.write(str(j))
         write.write('\r\n')
     write.close()

def script():
    [X,y]=read_file()
    #x_dat,y=preproccessing(X,y)
    #write_features(x_dat,y)
    return learn(X,y)

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
    return (full_data[:,0:-1],full_data[:,-1])
#script()


