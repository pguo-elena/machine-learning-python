import sys
import numpy as np
import pandas as pd
import math
from cvxopt import solvers, matrix
import sklearn.preprocessing as preprocessing
from scipy.spatial.distance import euclidean


class Experiment:

    def __init__(self):
        self.original_data = 0
        self.test_data = 0
        self.featurename = list()
        self.x_train = 0
        self.y_train = 0
        self.x_train_ori = 0
        self.y_train_ori = 0
        self.x_test = 0
        self.y_test = 0

    def onNB(self):
        clf = NaiveBayes()
        clf.fit(self.x_train,self.y_train)
        y_pred = clf.predict(self.x_test)
        #print metrics.log_loss(y_pred)
        #print metrics.confusion_matrix(self.y_test, y_pred)
        
        print self.logloss(y_pred)

    def onLR(self):
        clf = logesticRegression()
        clf.fit(self.x_train,self.y_train)
        y_pred = clf.predict(self.x_test)
        print self.logloss(y_pred)

    def onSVM(self):
        for i in range(self.y_train.shape[0]):
            if (self.y_train[i]==0):
                self.y_train[i]=-1
        for i in range(self.y_test.shape[0]):
            if (self.y_test[i]==0):
                self.y_test[i]=-1
        clf = SVM()
        clf.fit(True, self.x_train[:500], self.y_train[:500])
        y_pred = clf.predict(self.x_test[:500])
        self.y_test=self.y_test[:500]

        print self.logloss(y_pred)

    def onDT(self):
        clf = Decision_Tree(self.featurename,len(self.featurename),np.mean(self.x_train,axis = 0))
        mytree = clf.buildtree(self.x_train,self.y_train)
        y_pred = clf.predict(mytree,self.x_test)
        
        print self.logloss(y_pred)

    def load_data(self, trainDataSet, testDataSet):
        self.original_data = pd.read_csv(trainDataSet,
            names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", 
            "Martial Status", "Occupation", "Relationship", "Race", "Sex",
            "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',engine='python')
        #del original_data["Education"]
        self.test_data = pd.read_csv(testDataSet,
            names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", 
            "Martial Status", "Occupation", "Relationship", "Race", "Sex",
            "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',engine='python')
        #del test_data["Education"]

    def splitdata(self,part = 0):
        if (part):
            p = self.y_train_ori.shape[0]/20
            self.x_train = self.x_train_ori[:part*p]
            self.y_train = self.y_train_ori[:part*p]
        else:
            self.x_train = self.x_train_ori
            self.y_train = self.y_train_ori
        #print self.x_train

    def encode_data(self):
        encoded_data, encoders = self.number_encode_features(self.original_data)
        df = encoded_data[encoded_data.columns.difference(["Target"])]
        self.featurename = list(df.columns)
        self.x_train_ori = df.values
        self.y_train_ori = encoded_data["Target"].values
        encoded_data, encoders = self.number_encode_features(self.test_data)
        df = encoded_data[encoded_data.columns.difference(["Target"])]
        self.x_test = df.values
        self.y_test = encoded_data["Target"].values
        #print self.y_train

    def encode_data_scale(self):
        encoded_data, encoders = self.number_encode_features(self.original_data)
        scaler = preprocessing.StandardScaler()
        df = encoded_data[encoded_data.columns.difference(["Target"])]
        df = pd.DataFrame(scaler.fit_transform(df.astype("f64")), columns=df.columns)
        self.x_train_ori = df.values
        self.y_train_ori = encoded_data["Target"].values
        encoded_data, encoders = self.number_encode_features(self.test_data)
        df = encoded_data[encoded_data.columns.difference(["Target"])]
        df = pd.DataFrame(scaler.fit_transform(df.astype("f64")), columns=df.columns)
        self.x_test = df.values
        self.y_test = encoded_data["Target"].values

    def number_encode_features(self,df):
        #np.unique
        result = df.copy()
        encoders = {}
        for column in result.columns:
            if result.dtypes[column] == np.object:
                encoders[column] = preprocessing.LabelEncoder()
                result[column] = encoders[column].fit_transform(result[column])
        return result, encoders

    def logloss(self, ypred):
        epsilon = 1e-15
        ypred = np.maximum(epsilon, ypred)
        ypred = np.minimum(1-epsilon, ypred)
        res = sum(self.y_test*np.log(ypred) + \
            np.subtract(1,self.y_test)*np.log(np.subtract(1,ypred)))
        res = res * -1.0/len(self.y_test)
        return res

class logesticRegression:

    def __init__(self):
        self.theta = 0

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def fit(self, x_train, y_train, eta = 0.000001, epsilon = 0.001):
        row, colum = x_train.shape
        z = np.zeros((row, colum+1))
        for i in range(row):
            z[i][0] = 1
            for j in range(colum):
                z[i][j+1] = x_train[i][j]
        row, colum = z.shape
        theta_old = np.ones((1, colum))*0.000000001
        flag = True
        while flag:     
            #compute theta_new
            update = np.zeros((1, colum))
            for i in range(row):
                x_t = np.reshape(z[i], (colum, 1))
                #print np.dot(theta_old, x_t)
                yhat = self.sigmoid(np.dot(theta_old, x_t))
                update += eta*(yhat - y_train[i])*z[i]
            self.theta = theta_old - update
            
            diff = 0
            #compute difference of theta_new and theta_old
            for i in range(colum):
                diff += math.pow(self.theta[0][i] - theta_old[0][i], 2)      

            #judge condition
            #print diff
            if diff < epsilon:
                flag = False
                break
            else:
                theta_old = self.theta

    def predict(self, x_test):
        row, colum = x_test.shape
        z = np.zeros((row, colum+1))
        for i in range(row):
            z[i][0] = 1
            for j in range(colum):
                z[i][j+1] = x_test[i][j]
        row, colum = z.shape
        yhat = np.zeros((row))
        for i in range(row):
            x_t = np.reshape(z[i], (colum, 1))
            yhat[i] = self.sigmoid(np.dot(self.theta, x_t))<0.5
        return yhat

class NaiveBayes:

    def __init__(self):
        self.alpha = 0
        self.prior = 0

    def fit(self, train, test):
        trainm,trainn = train.shape
        for i in range(trainm):
            for j in range(trainn):
                train[i,j]=train[i,j]>0
        testm = test.size
        self.alpha = np.ones((2,trainn))
        count = np.zeros(2)
        for i in range(trainm):
            if (test[i]==0):
                count[0] += 1
                for j in range(trainn):
                    self.alpha[0,j]+=train[i,j]
            elif (test[i]==1):
                count[1] += 1
                for j in range(trainn):
                    self.alpha[1,j]+=train[i,j]
        self.alpha[0] = self.alpha[0]/count[0]
        self.alpha[1] = self.alpha[1]/count[1]
        self.prior = count/trainn

    def predict(self,test):
        testm,testn = test.shape
        for i in range(testm):
            for j in range(testn):
                test[i,j]=test[i,j]>0
        result = list()
        for i in range(testm):
            g1 = 0
            for j in range(testn):
                g1 += test[i,j]*np.log(self.alpha[0,j])+\
                (1-test[i,j])*np.log(self.alpha[1,j])+np.log(self.prior[0])
            g2 = 0
            for j in range(testn):
                g2 += test[i,j]*np.log(self.alpha[1,j])+\
                (1-test[i,j])*np.log(self.alpha[0,j])+np.log(self.prior[1])
            if (g1>g2):
                result.append(0)
            else:
                result.append(1)
        return np.array(result)

class SVM:
    def __init__(self):
        self.w = 0
        self.w0 = 0
        self.sv_index = 0
        self.sv_alpha = 0
        self.sv_x = 0
        self.sv_y = 0

    def fit(self,softmargin, x_train, y_train, c=1):
        row, colum = x_train.shape
        self.w = np.zeros((1, colum))
        q = -np.ones((row,1))
        b = np.zeros((1,1))
        y_train = np.reshape(y_train,(row,1))
        y_t = np.transpose(y_train)
        #print y_t.shape
        x_t = np.transpose(x_train)
        p = np.dot(y_train,y_t)*np.dot(x_train,x_t)
        A = y_t
    
        if softmargin == False:
            G = -np.identity(row)
            h = np.zeros((row, 1))
        else:
            upper_G = -np.identity(row)
            lower_G = np.identity(row)
            G = np.concatenate((upper_G, lower_G), axis = 0)
            upper_h = np.zeros((row, 1))
            lower_h = np.ones((row, 1))*c
            h = np.concatenate((upper_h, lower_h), axis = 0)
    
        #compute alpha
        solvers.options['show_progress'] = False
        p=p.astype(np.double)
        q=q.astype(np.double)
        G = G.astype(np.double)
        h = h.astype(np.double)
        A = A.astype(np.double)
        b = b.astype(np.double)
        #print "Parameters Ready"
        sol = solvers.qp(matrix(p),matrix(q),matrix(G),matrix(h),matrix(A),matrix(b))
        #print "Got Alpha"
        alpha = np.array(sol['x'])
        sv_index = list()
        sv_alpha = list()
        sv_x = list()
        sv_y = list()

        for i in range(row):
            if alpha[i][0] > 1e-4:
                sv_index.append(i)
                sv_alpha.append(alpha[i])
                sv_x.append(x_train[i])
                sv_y.append(y_train[i])

        self.sv_alpha = np.array(sv_alpha)
        self.sv_x = np.array(sv_x)
        self.sv_y = np.array(sv_y)
        self.sv_index = np.array(sv_index)

        #print "SV Got"

        #compute w
        for i in self.sv_index:
            for j in range(colum):
                self.w[0][j] += alpha[i][0]*y_train[i]*x_train[i][j]
    
        #compute w0
        total = 0
        wx = 0
        for i in self.sv_index:
            total += 1
            for j in range(colum):
                wx += self.w[0][j]*x_train[i][j]
            self.w0 += y_train[i] - wx
        self.w0 = self.w0/total

    def predict(self, x_test, kernel=2, q=2, sigma=0.1):
        row, colum = x_test.shape
        yhat = np.zeros(row)
        w_t = np.transpose(self.w)
        num = self.sv_alpha.shape[0]
        
        if kernel == 0:
            for i in range(row):
                yhat[i] =  np.dot(x_test[i], w_t) + self.w0
                if yhat[i] < 0:
                    yhat[i] = -1
                else:
                    yhat[i] = 1
        elif kernel == 1:  
            for i in range(row):
                x_t = np.transpose(x_test[i])
                for j in range(num):
                    yhat[i] += self.sv_alpha[j]*self.sv_y[j]*(np.dot(self.sv_x[j],x_t)**q)
                yhat[i] = yhat[i] + self.w0
                if yhat[i] < 0:
                    yhat[i] = -1
                else:
                    yhat[i] = 1

        elif kernel == 2:
            for i in range(row):
                for j in range(num):
                    yhat[i] += self.sv_alpha[j]*self.sv_y[j]*(np.exp(euclidean(self.sv_x[j], x_test[i])/((-2)*(sigma**2))))
                if yhat[i] < 0:
                    yhat[i] = -1
                else:
                    yhat[i] = 1

        return yhat

class Decision_Tree:
    def __init__(self, fname, fnum, args):
        self.feaname=fname[:]
        self.feanamecopy=fname[:]
        self.args = args
    
    def calentropy(self,ytrain):
        n = ytrain.size
        count = {} 
        for curlabel in ytrain:
            if curlabel not in count.keys():
                count[curlabel] = 0
            count[curlabel] += 1
        #print count
        entropy = 0
        for key in count:
            t = float(count[key])/n
            entropy -= t * np.log2(t)
        return entropy
    
    def splitdata(self,data,splitfeat):
        pivot = self.args[splitfeat]
        less = []
        greater = []
        for i in range(len(data)):
            if data[i][splitfeat] < pivot:
                less.append(i)
            else:
                greater.append(i)
        return less,greater
    
    def appdata(self,data,label,splitidx,fea_idx):
        datal = []
        datag = []
        for i in splitidx[0]:
            datal.append(np.append(data[i][:fea_idx],data[i][fea_idx+1:]))
        for i in splitidx[1]:
            datag.append(np.append(data[i][:fea_idx],data[i][fea_idx+1:]))
        labell = label[splitidx[0]]
        labelg = label[splitidx[1]]
        return datal,datag,labell,labelg
    
    def choosenode(self,data,label):
        base_entropy = self.calentropy(label)
        best_gain = -32768
        for i in range(len(data[0])):
            data_less, data_greater = self.splitdata(data,i)
            prob_less = float(len(data_less))/len(label)
            prob_greater = float(len(data_greater))/len(label)
            
            entropy = prob_less * self.calentropy(label[data_less]) \
                + prob_greater * self.calentropy(label[data_greater])
            
            info = base_entropy - entropy
            if (info>best_gain):
                result = i
                best_gain = info
        return result
    
    
    def buildtree(self,data,label):
        if label.size==0:
            return 0
        if label.tolist().count(label[0])==label.size:
            return label[0]
        if len(self.feanamecopy)==0:
            cnt = {}
            for i in label:
                if i not in cnt.keys():
                    cnt[i] = 0
                cnt[i] += 1
            return max(cnt, key=cnt.get)
        
        bestnode = self.choosenode(data,label)
        #print bestnode,len(data[0])
        fname = self.feanamecopy[bestnode]
        #print fname
        nodedict = {fname:{}}
        del(self.feanamecopy[bestnode])
        spoint = self.splitdata(data,bestnode)
        data_less,data_greater,label_less,label_greater = self.appdata(data,label,spoint,bestnode)
        
        nodedict[fname]["<"] = self.buildtree(data_less,label_less)
        nodedict[fname][">"] = self.buildtree(data_greater,label_greater)
        return nodedict
    
    def classify(self,mytree,testdata):
        if type(mytree).__name__ != 'dict':
            return mytree
        fname = mytree.keys()[0] 
        findex = self.feaname.index(fname)
        nextbranch = mytree[fname]
        
        if testdata[findex]>self.args[findex]:
            nextbranch = nextbranch[">"]
        else:
            nextbranch = nextbranch["<"]
        return self.classify(nextbranch,testdata)

    def predict(self, mytree, xtest):
        ypred = list()
        for i in range(len(xtest)):
            ypred.append(self.classify(mytree, xtest[i]))
        return np.array(ypred)

if __name__=="__main__":
    inst = Experiment()
    inst.load_data("../data/adultMod.data", "../data/adultMod.test")
    inst.encode_data_scale()
    for i in range(1,21):
        inst.splitdata(i)
        inst.onDT()

