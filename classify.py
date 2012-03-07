import numpy
import io
import os
import mlpy
import random
paper_theta=[-0.274208077364182,	-1.19702842106444,0.131548956323387,	0.479300305088592,	-4.62317179538485,	2.76767389462995,	-0.602063611626551,	-0.471415566064060,	0.839184445514958,	-0.00951651617889351,	-0.183643075791352]
rock_theta=[-1.50086345729911,	1.02170469221557,	1.32845253378158,	0.743290065372132,	3.56216260278406,	-0.719749544797365,	-4.09732517490453,	2.24331283975708,	0.243933641037109,	1.90148153447562,	-1.58640711819051,]
scissors_theta=[-2.68358663497958,	2.57142732926036,	-1.16225707922569,	-0.821983171352408,	1.76839157489140,	-1.48458274872344,	2.88747222454093,	-1.61516200934215,	-1.36761600014807,	-2.25620393145480,	2.71852319751607]
pca_matrix=[[-0.0612518067173682,	0.660674896446432,	-0.277388755196180,	0.286111945032877]
,[-0.104470926852773,	0.673339792663848,	-0.125449312046432,	-0.354458906183283]
,[-0.382238219749070,	0.210283903997366,	0.572283444589444,	0.0949386892218484]
,[-0.384832897823110,	-0.112753500025810,	-0.397731579265045,	0.636114360127679]
,[-0.532267950707037,	-0.0704233062079431,	0.130265409151969,	-0.414239052522610]
,[-0.575181521200229,	-0.0966646000262674,	0.119020374639945,	0.137854187096441]
,[0.277468326214780,	0.197195450801897,	0.624887407539152,	0.433873524578528]
]
mu=[8.65048773353768e-16, -4.49582500857323e-17,-9.25185853854297e-17,1.85037170770859e-17,4.16333634234434e-17,-1.29526019539602e-16,2.77555756156289e-17]

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
    
    #data_cov=numpy.cov(data)
    #
    #[E, lamb, _temp]=numpy.linalg.svd(numpy.cov(data.T))
    #components=E[:,0:4]
    #data_pca=numpy.dot(data,components)
    #
    #l=data_pca[:,0]*data_pca[:,1]
    #print(numpy.size(l[:,newaxis],0))
    #print(numpy.size(l[:,newaxis],1))
    #X=numpy.column_stack((data_pca))
    
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
script()



