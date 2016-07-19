from __future__ import division
import numpy as np
import math
from sklearn.cross_validation import KFold
from sklearn.datasets import fetch_mldata
import os
from pylab import *
#from sklearn.neural_network import MLPClassifier

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def reconstructXa(x):
	row, colum = x.shape
	x_new = np.zeros((row, colum+1))
	for i in range(row):
		x_new[i][0] = 1
		for j in range(colum):
			x_new[i][j+1] = x[i][j]
	return x_new

def twoClassLR(x_train, y_train, eta, epsilon):
	row, colum = x_train.shape
	theta_old = np.ones((1, colum))*0.1
	#theta_new = np.zeros((1, colum))
	
	flag = True
	while flag:		
		#compute theta_new
		update = np.zeros((1, colum))
		for i in range(row):
			x_t = np.reshape(x_train[i], (colum, 1))
			yhat = sigmoid(np.dot(theta_old, x_t))
			update += eta*(yhat - y_train[i])*x_train[i]
		theta_new = theta_old - update
		
		diff = 0
		#compute difference of theta_new and theta_old
		for i in range(colum):
			diff += math.pow(theta_new[0][i] - theta_old[0][i], 2)		

		#judge condition
		#print diff
		if diff < epsilon:
			flag = False
			break
		else:
			theta_old = theta_new

	return theta_new

def predict(theta, x_test, y_test, c):
	#print theta
	row, colum = x_test.shape
	for i in range(row):
		#print x_test[i]
		x_t = np.reshape(x_test[i], (colum, 1))
		yhat = sigmoid(np.dot(theta, x_t))
		#print yhat
		if yhat < 0.5:
			if y_test[i] == 0:
				c[0][0] += 1
			else:
				c[0][1] += 1
		else:
			if y_test[i] == 0:
				c[1][0] += 1
			else:
				c[1][1] += 1
	return c

		
def prob1_a():
	#load data
	fname = "heart.dat"
	dataset = np.loadtxt(fname)
	x = dataset[:,0:-1]
	y = dataset[:, -1]-1

	#add 1 to frist colum of each row of x
	x = reconstructXa(x)

	#set user specified parameters
	eta = 0.000005
	epsilon = 0.001

	#confusion matrix
	c = np.zeros((2, 2))

	#do kfold cross validation
	kf = KFold(x.shape[0], n_folds = 2, shuffle = True, random_state = 0)
	for train_index, test_index in kf:
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#train theta
		theta = twoClassLR(x_train, y_train, eta, epsilon)

		#test perpormance
		c = predict(theta, x_test, y_test, c)
	print c

	a = c.shape[0]
	precision = np.ones(a)
	recall = np.ones(a)
	f_measure = np.ones(a)
	accuracy = 0
	#compute precision
	nume = 0
	for i in range(c.shape[0]):
		nume = c[i][i]
		deno = 0
		for j in range(c.shape[1]):
			deno += c[i][j]
		precision[i] = nume/deno

	#compute recall
	for j in range(c.shape[1]):
		nume = c[j][j]
		deno = 0
		for i in range(c.shape[0]):
			deno += c[i][j]
		recall[j] = nume/deno
	
	#compute f-measure
	for i in range(c.shape[0]):
		f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
	
	#compute accuracy
	nume = 0
	deno = 0
	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			if i == j:
				nume += c[i][j]
			deno += c[i][j]
	accuracy = nume/deno
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy

def reconstructXb(x):
	row, colum = x.shape
	x_new = np.zeros((row, 2*colum+1))
	for i in range(row):
		#print x[i]
		x_new[i][0] = 1
		for j in range(colum):
			x_new[i][j+1] = x[i][j]
		for k in range(colum):
			x_new[i][colum+1+k] = math.pow(x[i][k],0.5)
	return x_new

def prob1_b():
	#load data
	fname = "heart.dat"
	dataset = np.loadtxt(fname)
	x = dataset[:,0:-1]
	y = dataset[:, -1]-1

	#add 1 to frist colum of each row of x and use non-linear combinations of inputs
	x = reconstructXb(x)

	#set user specified parameters
	eta = 0.000001
	epsilon = 0.00001

	#confusion matrix
	c = np.zeros((2, 2))
	
	#do kfold cross validation
	kf = KFold(x.shape[0], n_folds = 2, shuffle = True, random_state = 0)
	for train_index, test_index in kf:
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#train theta
		theta = twoClassLR(x_train, y_train, eta, epsilon)

		#test perpormance
		c = predict(theta, x_test, y_test, c)
	print c

	a = c.shape[0]
	precision = np.ones(a)
	recall = np.ones(a)
	f_measure = np.ones(a)
	accuracy = 0
	#compute precision
	nume = 0
	for i in range(c.shape[0]):
		nume = c[i][i]
		deno = 0
		for j in range(c.shape[1]):
			deno += c[i][j]
		precision[i] = nume/deno

	#compute recall
	for j in range(c.shape[1]):
		nume = c[j][j]
		deno = 0
		for i in range(c.shape[0]):
			deno += c[i][j]
		recall[j] = nume/deno
	
	#compute f-measure
	for i in range(c.shape[0]):
		f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
	
	#compute accuracy
	nume = 0
	deno = 0
	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			if i == j:
				nume += c[i][j]
			deno += c[i][j]
	accuracy = nume/deno
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy

def kClassLR(x_train, y_train, eta, epsilon):
	row, colum = x_train.shape
	theta_old = np.ones((3, colum))*0.005
	theta_new = np.ones((3, colum))*0.01
	
	flag = True
	while flag:	
		#print 1	
		#print theta_old
		#compute theta_new
		for j in range(theta_old.shape[0]):
			update = np.zeros((1, colum))
			for i in range(row):
				x_t = np.reshape(x_train[i], (colum, 1))
				#print x_t
				denom = math.exp(np.dot(theta_old[0], x_t)) + math.exp(np.dot(theta_old[1], x_t)) \
						+ math.exp(np.dot(theta_old[2], x_t))
				nume = math.exp(np.dot(theta_old[j], x_t))
				yhat = nume/denom
				indicator = 0
				if y_train[i] == j:
					indicator = 1
				else:
					indicator = 0
				update += eta*(yhat - indicator)*x_train[i]

			theta_new[j] = theta_old[j] - update
		#print theta_new
		
		diff = 0
		#compute difference of theta_new and theta_old
		for i in range(theta_new.shape[0]):
			for j in range(theta_new.shape[1]):
				diff += math.pow(theta_new[i][j] - theta_old[i][j], 2)

		#judge condition
		#print diff
		if diff < epsilon:
			flag = False
			break
		else:
			theta_old = theta_new

	return theta_new

def predictKclass(theta, x_test, y_test, c):
	row, colum = x_test.shape
	pred_val = np.zeros((3,1))
	for i in range(row):
		x_t = np.reshape(x_test[i], (colum, 1))
		#compute probability of x_test[i] for each class
		pred_val[0] = np.dot(theta[0], x_t)
		pred_val[1] = np.dot(theta[1], x_t)
		pred_val[2] = np.dot(theta[2], x_t)

		#find maximum probability of calss as the predict result
		max_val = 0
		yhat = 0
		for j in range(pred_val.shape[0]):
			if pred_val[j] > max_val:
				max_val = pred_val[j]
				yhat = j
		k = y_test[i]
		c[yhat][k] += 1

	return c 

def prob1_c():
	#load data
	fname = "wine.data"
	datasets = np.loadtxt(fname, delimiter = ',')
	x = datasets[: , 1:]
	y = datasets[:, 0] - 1

	#add 1 to frist colum of each row of x
	x = reconstructXa(x)

	#set user specified parameters
	eta = 0.0000001
	epsilon = 0.0000000001

	#confusion matrix
	c = np.ones((3, 3))

	#do kfold cross validation
	kf = KFold(x.shape[0], n_folds = 2, shuffle = True, random_state = 0)
	for train_index, test_index in kf:
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#train theta
		theta = kClassLR(x_train, y_train, eta, epsilon)

		#test perpormance
		c = predictKclass(theta, x_test, y_test, c)
	
	print c

	a = c.shape[0]
	precision = np.ones(a)
	recall = np.ones(a)
	f_measure = np.ones(a)
	accuracy = 0
	#compute precision
	nume = 0
	for i in range(c.shape[0]):
		nume = c[i][i]
		deno = 0
		for j in range(c.shape[1]):
			deno += c[i][j]
		precision[i] = nume/deno

	#compute recall
	for j in range(c.shape[1]):
		nume = c[j][j]
		deno = 0
		for i in range(c.shape[0]):
			deno += c[i][j]
		recall[j] = nume/deno
	
	#compute f-measure
	for i in range(c.shape[0]):
		f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
	
	#compute accuracy
	nume = 0
	deno = 0
	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			if i == j:
				nume += c[i][j]
			deno += c[i][j]
	accuracy = nume/deno
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy

def softMax(k, a, x):
    denom = math.exp(np.dot(a[0], x)) + math.exp(np.dot(a[1], x)) \
                        + math.exp(np.dot(a[2], x))
    nume = math.exp(np.dot(a[k], x))
    return nume/denom


def kClassMLP(h, x_train, y_train, eta, epsilon):
    row, colum = x_train.shape
    w_old = np.ones((h, colum))*0.005
    w_new = np.ones((h, colum))*0.005
    v_old = np.ones((3, h+1))*0.005
    v_new = np.ones((3, h+1))*0.005
    z = np.ones((1, h+1))
    yhat = np.zeros(4)
    pred_val = np.zeros((4, row))
    result = np.zeros(row)

    flag = True
    while flag: 
        for i in range(row):
            x_t = np.reshape(x_train[i], (colum, 1))
            #compute z
            for j in range(h):
                z[0][j+1] = sigmoid(np.dot(w_old[j], x_t))
            #compute yhat
            z_t = np.reshape(z, (h+1, 1))
            for k in range(3):
                yhat[k+1] = softMax(k, v_old, z_t)
            #update v
            indicator = np.zeros(4)
            for l in range(3):
                if y_train[i] == l+1:
                    indicator[l+1] = l+1
                else:
                    indicator[l+1] = 0
                v_new[l] = v_old[l] - eta*(yhat[l+1]-indicator[l+1])
            #update w
            for p in range(h):
                part = 0
                for q in range(3):
                    part += (yhat[q+1]-indicator[q+1])*v_new[q][p+1]
                w_new[p] = w_old[p] - eta*part*z[0][p+1]*(1-z[0][p+1])*x_train[i]
            #compute objective function
            obj = 0
            for s in range(row):
                for t in range(3):
                    pred_val[t+1][s] = softMax(t, v_old, z_t)
                max_val = 0
                for r in range(3):
                    if pred_val[r+1][s] > max_val:
                        max_val = pred_val[r+1][s]
                        result[s] = r+1
                obj += 0.5*((result[s]-y_train[s])**2)
            print obj
            if obj < epsilon:
                flag = False
                break
            else:
                v_old = v_new
                w_old = w_new

    return w_new, v_new

def predictKclassMLP(h, w, v, x_test, y_test, c):
    row, colum = x_test.shape
    z = np.ones((1, h+1))
    pred_val = np.zeros((3,1))
    for i in range(row):
        x_t = np.reshape(x_test[i], (colum, 1))
        #compute probability of x_test[i] for each class
        for k in range(1, h+1):
            z[0][k] = sigmoid(np.dot(w[k-1], x_t))
        z_t = np.reshape(z, (h+1, 1))
        pred_val[0] = np.dot(v[0], z_t)
        pred_val[1] = np.dot(v[1], z_t)
        pred_val[2] = np.dot(v[2], z_t) 

        #find maximum probability of calss as the predict result
        max_val = 0
        yhat = 0
        for j in range(3):
            if pred_val[j] > max_val:
                max_val = pred_val[j]
                yhat = j
        k = y_test[i]
        c[yhat-1][k-1] += 1

    return c 

def prob2_b():
    #load dataset
    mnist = fetch_mldata('MNIST original')
    mnist.data.shape
    mnist.target.shape
    np.unique(mnist.target)

    x, y = mnist.data / 255., mnist.target
    #extract part of datasat use for training and testing
    x0 = x[:100]
    y0 = y[:100]
    x1 = x[5923:6023]
    y1 = y[5923:6023]
    x2 = x[12665:12765]
    y2 = y[12665:12765]
    x = np.concatenate((x0, x1, x2), axis = 0)
    y = np.concatenate((y0, y1, y2), axis = 0)
    y = y+1
    #confusion matrix
    c = np.ones((3, 3))
    #add 1 to frist colum of each row of x
    x = reconstructXa(x)
    #set user specified parameters
    eta = 0.0000001
    epsilon = 60
    #specify number of hidden units
    h = 50

    #do kfold cross validation
    kf = KFold(x.shape[0], n_folds = 2, shuffle = True, random_state = 0)
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #train parameters
        w, v = kClassMLP(h, x_train, y_train, eta, epsilon)

        #test perpormance
        c = predictKclassMLP(h, w, v, x_test, y_test, c)
    
    print c

    a = c.shape[0]
    precision = np.ones(a)
    recall = np.ones(a)
    f_measure = np.ones(a)
    accuracy = 0
    #compute precision
    nume = 0
    for i in range(c.shape[0]):
        nume = c[i][i]
        deno = 0
        for j in range(c.shape[1]):
            deno += c[i][j]
        precision[i] = nume/deno

    #compute recall
    for j in range(c.shape[1]):
        nume = c[j][j]
        deno = 0
        for i in range(c.shape[0]):
            deno += c[i][j]
        recall[j] = nume/deno
    
    #compute f-measure
    for i in range(c.shape[0]):
        f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
    
    #compute accuracy
    nume = 0
    deno = 0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if i == j:
                nume += c[i][j]
            deno += c[i][j]
    accuracy = nume/deno
    print "precision:", precision
    print "recall:", recall
    print "f-measure:", f_measure
    print "accuracy:", accuracy

def skMLP(x, y, c):
	kf = KFold(x.shape[0], n_folds = 2, shuffle = True, random_state = 0)
	for train_index, test_index in kf:
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf = MLPClassifier(activation = 'logistic', algorithm = 'adam')
		clf.fit(x_train, y_train)
		yhat = clf.predict(x_test)
		for i in range(yhat.shape[0]):
			k = y_test[i]
			c[yhat][k] += 1
	print c
	a = c.shape[0]
	precision = np.ones(a)
	recall = np.ones(a)
	f_measure = np.ones(a)
	accuracy = 0
	#compute precision
	nume = 0
	for i in range(c.shape[0]):
		nume = c[i][i]
		deno = 0
		for j in range(c.shape[1]):
			deno += c[i][j]
		precision[i] = nume/deno

	#compute recall
	for j in range(c.shape[1]):
		nume = c[j][j]
		deno = 0
		for i in range(c.shape[0]):
			deno += c[i][j]
		recall[j] = nume/deno
	
	#compute f-measure
	for i in range(c.shape[0]):
		f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
	
	#compute accuracy
	nume = 0
	deno = 0
	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			if i == j:
				nume += c[i][j]
			deno += c[i][j]
	accuracy = nume/deno
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy

if __name__=="__main__":
	#prob1_a()
	#prob1_b()
	#prob1_c()
	prob2_b()
