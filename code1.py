from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
import time
from scipy.spatial.distance import euclidean


#load wohle data file, split feature and label
def loadData(fname):
	datasets = np.loadtxt(fname)
	x = datasets[:, 0]
	y = datasets[:, -1]
	return x, y

#plot dataseti	
def plotSvarData(i, x, y):
	plt.subplot(2, 2, i)
	plt.plot(x, y, 'g.')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title("datasets" + str(i))

#plot single variable linear model regression of dataseti
def plotSvarLinearModel(i, x, y):
	plt.subplot(2, 2, i)
	theta = svarLinearModel(x, y)
	yhat = x*theta[1] + theta[0]
	plt.plot(x, yhat, 'b')

#plot single variable linear model used ready make function of dataseti
def plotSvarLinearFun(i, x, y):
	regr = linear_model.LinearRegression()
	x = np.reshape(x, (x.shape[0], 1))
	y = np.reshape(y, (y.shape[0], 1))
	regr.fit(x, y)
	yhat = regr.predict(x)
	plotSvarData(i, x, y)
	plt.plot(x, yhat, 'r')

#compute parameters of single variable linear model(fit data)
def svarLinearModel(x, y):
	m = x.shape[0]
	sum_x = 0
	sum_xsq = 0
	sum_y = 0
	sum_xy = 0
	for i in range(m):
		sum_x += x[i]
		sum_xsq += x[i]**2
		sum_y += y[i]
		sum_xy += x[i]*y[i]
	A = np.array([(m, sum_x), (sum_x, sum_xsq)])
	b = np.array([(sum_y),(sum_xy)])
	theta = np.linalg.solve(A,b)
	return theta

#predict label for given feature using parameters
def svarLinearPredict(x, theta):
	yhat = x*theta[1] + theta[0]
	return yhat

#compute RSE for given pridiction and real label	
def computeRSE(y, yhat):
	m = y.shape[0]
	sum_error = 0
	for i in range(m):
		sum_error += ((yhat[i] - y[i])**2) /(y[i]**2)
	rse = sum_error / m
	return rse

#using ready made function to compute parameters and compute error for single variable linear model
def callLinearFun(x_train, y_train, x_test, y_test):
	regr = linear_model.LinearRegression()
	x_train = np.reshape(x_train, (x_train.shape[0], 1))
	y_train = np.reshape(y_train, (y_train.shape[0], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], 1))
	y_test = np.reshape(y_test, (y_test.shape[0], 1))
	regr.fit(x_train, y_train)
	yhat_train = regr.predict(x_train)
	yhat = regr.predict(x_test)
	train_rse = computeRSE(y_train, yhat_train)
	test_rse = computeRSE(y_test, yhat)
	return train_rse[0], test_rse[0]
	'''
	sum_error = 0
	for j in range(x.shape[0]):
		sum_error += ((yhat[j] - y[j])**2) /(y[j]**2)
	rse = sum_error / x.shape[0]
	return rse[0]
	'''
#plot single variable polynomial model for dataseti
def plotSvarPolyModel(i, x, y, degree):
	theta = svarPolyModel(x, y, degree)
	z = np.zeros((x.shape[0], degree+1))
	for k in range(x.shape[0]):
		for j in range(degree+1):
			z[k][j] = x[k]**j
	yhat = np.dot(z, theta)
	plt.subplot(2, 2, i)
	plotSvarData(i, x, y)
	plt.plot(x, yhat, 'bo', alpha = 0.3)

#plot single variable polynomial model using ready made function
def plotSvarPolyFun(i, x, y, degree):
	regr = linear_model.LinearRegression()
	x = np.reshape(x, (x.shape[0], 1))
	y = np.reshape(y, (y.shape[0], 1))
	poly = PolynomialFeatures(degree)
	x_poly = poly.fit_transform(x)
	regr.fit(x_poly, y)
	yhat = regr.predict(x_poly)
	plotSvarData(i, x, y)
	plt.plot(x, yhat, 'ro', alpha = 0.3)

#compute parameters of single variable polynomial model(fit data)
def svarPolyModel(x, y, degree):
	#transform x to z
	z = np.zeros((x.shape[0], degree+1))
	for i in range(x.shape[0]):
		for j in range(degree+1):
			z[i][j] = x[i]**j
	theta = np.dot(np.linalg.pinv(z), y)
	return theta

#predict label for given feature and parameters
def svarPolyPredict(x, degree, theta):
	z = np.zeros((x.shape[0], degree+1))
	for i in range(x.shape[0]):
		for j in range(degree+1):
			z[i][j] = x[i]**j
	yhat = np.dot(z, theta)
	return yhat

#compute parameters, make prediction, and compute RSE for ready made function of single variable polynomial model
def callPolyFun(x_train, y_train, x_test, y_test, degree):
	regr = linear_model.LinearRegression()
	x_train = np.reshape(x_train, (x_train.shape[0], 1))
	x_test = np.reshape(x_test, (x_test.shape[0], 1))
	y_train = np.reshape(y_train, (y_train.shape[0], 1))
	y_test = np.reshape(y_test, (y_test.shape[0], 1))
	#transform x to z	
	poly = PolynomialFeatures(degree)
	x_train_poly = poly.fit_transform(x_train)
	x_test_poly = poly.fit_transform(x_test)
	regr.fit(x_train_poly, y_train)
	yhat_train = regr.predict(x_train_poly)
	yhat = regr.predict(x_test_poly)
	train_rse = computeRSE(y_train, yhat_train)
	test_rse = computeRSE(y_test, yhat)
	return train_rse[0], test_rse[0]
	'''
	sum_error = 0
	for j in range(x.shape[0]):
		sum_error += ((yhat[j] - y[j])**2) /(y[j]**2)
	rse = sum_error / x.shape[0]
	return rse[0]
	'''

#generate 10 cross validation
def crossValidation(num, fname, percent):
	datasets = np.loadtxt(fname)
	total = datasets.shape[0]
	part = total * percent
	datasets = datasets[0:200,:]
	total = datasets.shape[0]
	'''
	x_test = datasets[num*(total /10):(num+1)*(total /10),0]
	x_train = np.hstack((datasets[0:num*(total /10), 0], datasets[(num+1)*(total /10): , 0]))
	y_test = datasets[num*(total /10):(num+1)*(total /10),-1]
	y_train = np.hstack((datasets[0:num*(total /10), -1], datasets[(num+1)*(total /10): , -1]))
	'''
	x_test = datasets[num*(total /10):(num+1)*(total /10),0:-1]
	x_train = np.concatenate((datasets[0:num*(total /10), 0:-1], datasets[(num+1)*(total /10): , 0:-1]), axis = 0)
	y_test = datasets[num*(total /10):(num+1)*(total /10),-1]
	y_train = np.hstack((datasets[0:num*(total /10), -1], datasets[(num+1)*(total /10): , -1]))
	
	return x_test, x_train, y_test, y_train

def mappingToHighDimension(x, degree):
	z = np.ones((x.shape[0], 1))
	for degree in range(1, degree+1):
		z = np.append(z, x[:,0:x.shape[0]]**degree, axis = 1)
	return z

def mvarPolyModel(z, y):
	theta = np.dot(np.linalg.pinv(z), y)
	return theta

def mvarPredict(z, theta):
	yhat = np.dot(z, theta)
	return yhat

def computeMSE(y, yhat):
	m = y.shape[0]
	sum_error = 0
	for i in range(m):
		sum_error += ((yhat[i] - y[i])**2)
	mse = sum_error / m
	return mse

def judge(z_i, theta_old, theta_new, y_i, j_old, j_new, i):
	j_old[i] = (np.dot(z_i, theta_old) -y_i)**2
	#print i
	#print "theta_old", theta_old
	#print "theta_new", theta_new
	
	j_new[i] = (np.dot(z_i, theta_new) -y_i)**2
	#print "j_old[i]", j_old[i]
	#print "j_new[i]", j_new[i]
	jNew = np.sum(j_new)
	jOld = np.sum(j_old)
	#print "jOld", jOld
	#print "jNew", jNew

	return abs(jNew - jOld)

def iterativeSol(z, y):
	#stochastic gradient descent
	n = z.shape[1]
	m = z.shape[0]
	theta_old = np.ones((n, 1)) 
	theta_new = np.ones((n, 1))
	j_old = np.zeros((m, 1))
	j_new = np.zeros((m, 1))
	
	#compute learning rate using newton's method
	#eta = np.linalg.inv(np.dot(np.transpose(z),z))
	eta = 0.0001
	epsilon = 0.001
	flag = True
	k=0
	while flag:
		for i in range(m):
			z_i = np.reshape(z[i], (1, z[i].shape[0]))
			theta_new = theta_old - np.dot(eta,(np.dot(z_i, theta_old)-y[i])* np.transpose(z_i))
			diff = judge(z_i, theta_old, theta_new, y[i], j_old, j_new, i)
			#print "diff", diff
			if  diff< epsilon:
				flag = False
				break
			k+=1
			if (k%100000==0):
				epsilon *= 10
				k=0
	return theta_new


def kernelFun(x_datai, x_dataj):
	sigma = 1
	k = np.exp(euclidean(x_datai, x_dataj)/((-2)*(sigma**2)))
	return k

def gramMatrix(x_data):
	m = x_data.shape[0]
	gram = np.ones((m,m))
	for i in range(m):
		for j in range(m):
			gram[i][j] = kernelFun(x_data[i], x_data[j])
	return gram

def solveDual(x_data, x_feature, y):
	gram = gramMatrix(x_data)
	alpha = np.linalg.solve(gram, y)
	x_feature = np.transpose(x_feature)
	yhat = np.dot(np.dot(np.transpose(alpha), x_data), x_feature)
	return yhat



if __name__=="__main__":

	'''
	----------------------------question 1------------------------------------------------
	
	'''
	#set degree for polynomial model
	degree = 9
	# save ten cross validation, ten series of errors
	svarLRse_train = np.zeros((4, 10))
	svarLRse_train1 = np.zeros((4,10))
	svarLRse_test = np.zeros((4,10))
	svarLRse_test1 = np.zeros((4,10))
	svarPRse_train = np.zeros((4,10))
	svarPRse_train1 = np.zeros((4,10))
	svarPRse_test = np.zeros((4,10))
	svarPRse_test1 = np.zeros((4,10))

	# save mean error of ten cross validation
	svarL_trainError = np.zeros(4)
	svarL_testError = np.zeros(4)
	svarL_trainError1 = np.zeros(4)
	svarL_testError1 = np.zeros(4)
	svarP_trainError = np.zeros(4)
	svarP_testError = np.zeros(4)
	svarP_trainError1 = np.zeros(4)
	svarP_testError1 = np.zeros(4)

	for i in range(4):
		fname = "svar-set" + str(i+1) + ".dat"
		for num in range(10):
			x_test, x_train, y_test, y_train = crossValidation(num, fname, 1)
			theta_linear = svarLinearModel(x_train, y_train)
			#get svar linear model training error
			yhat_train = svarLinearPredict(x_train, theta_linear)
			svarLRse_train[i][num] = computeRSE(y_train, yhat_train)
			#get svar linear model testing error
			yhat = svarLinearPredict(x_test, theta_linear)
			svarLRse_test[i][num] = computeRSE(y_test, yhat)
			#get svar linear model ready made function training and testing error
			svarLRse_train1[i][num], svarLRse_test1[i][num] = callLinearFun(x_train, y_train, x_test, y_test)

			theta_poly = svarPolyModel(x_train, y_train, degree)
			#get svar polynomial model training error
			yhat_train = svarPolyPredict(x_train, degree, theta_poly)
			svarPRse_train[i][num] = computeRSE(y_train, yhat_train)
			#get svar polynomial model testing error
			yhat = svarPolyPredict(x_test, degree, theta_poly)
			svarPRse_test[i][num] = computeRSE(y_test, yhat)
			#get svar polynomial model ready made function training and testing error
			svarPRse_train1[i][num], svarPRse_test1[i][num] = callPolyFun(x_train, y_train, x_test, y_test, degree)

		#compute mean error
		for j in range(10):
			svarL_trainError[i] += (svarLRse_train[i][j] /10)
			svarL_testError[i] += (svarLRse_test[i][j] /10)
			svarL_trainError1[i] += (svarLRse_train1[i][j] /10)
			svarL_testError1[i] += (svarLRse_test1[i][j] /10)
			svarP_trainError[i] += (svarPRse_train[i][j] /10)
			svarP_testError[i] += (svarPRse_test[i][j] /10)
			svarP_trainError1[i] += (svarPRse_train1[i][j] /10)
			svarP_testError1[i] += (svarPRse_test1[i][j] /10)

	print "single variable linear model training error:\n",svarL_trainError 
	print "single variable linear model testing error:\n", svarL_testError 
	print "single variable linear model ready made function training error:\n",svarL_trainError1 
	print "single variable linear model ready made function testing error:\n",svarL_testError1 
	print "single variable polynomial model training error:\n",svarP_trainError 
	print "single variable polynomial model testing error:\n",svarP_testError 
	print "single variable polynomial model ready made function training error:\n",svarP_trainError1 
	print "single variable polynomial model ready made function testing error:\n",svarP_testError1 
	
	#plot figure
	for k in range(4):
		fname = "svar-set" + str(k+1) + ".dat"
		x, y = loadData(fname)
		plotSvarData(k+1, x, y)
		plotSvarLinearModel(k+1, x, y)
	plt.show()
	
	for k in range(4):
		fname = "svar-set" + str(k+1) + ".dat"
		x, y = loadData(fname)	
		plotSvarLinearFun(k+1, x, y)
	plt.show()
	
	for k in range(4):
		fname = "svar-set" + str(k+1) + ".dat"
		x, y = loadData(fname)	
		plotSvarData(k+1, x, y)
		plotSvarPolyModel(k+1, x, y, degree)
	plt.show()
	
	for k in range(4):
		fname = "svar-set" + str(k+1) + ".dat"
		x, y = loadData(fname)
		plotSvarPolyFun(k+1, x, y, degree)
	plt.show()

	'''
	
	-------------------------------question 2------------------------------------------------------
	'''
	#set degree for high dimension mapping
	degree = 2
	# save ten cross validation, ten series of errors
	mvarLMse_train = np.zeros((4, 10))
	mvarLMse_test = np.zeros((4,10))
	
	mvarIteMse_train = np.zeros((4,10))
	mvarIteMse_test = np.zeros((4,10))
	
	mvarDualMse_train = np.zeros((4,10))
	mvarDualMse_test = np.zeros((4,10))

	# save mean error of ten cross validation
	mvarL_trainError = np.zeros(4)
	mvarL_testError = np.zeros(4)
	
	mvarIte_trainError = np.zeros(4)
	mvarIte_testError = np.zeros(4)
	
	mvarDual_trainError = np.zeros(4)
	mvarDual_testError = np.zeros(4)

	a=0
	b=0
	c=0

	for i in range(4):
		print "Dataset "+str(i+1)
		fname = "mvar-set" + str(i+1) + ".dat"
		for num in range(10):
			#print num
			#get data
			x_test, x_train, y_test, y_train = crossValidation(num, fname, 0.1)
			#perform linear regression in the higher dimensional
			#for trainning error
			z = mappingToHighDimension(x_train, degree)
			#solve using explicit solution
			#get mvar explicit solution training error
			t = time.time()
			theta_explicit = mvarPolyModel(z, y_train)
			yhat_train = mvarPredict(z, theta_explicit)
			mvarLMse_train[i][num] = computeMSE(y_train, yhat_train)
			a += time.time()-t
			#print "step0"

			#solve using iterative solution
			#get mvar iterative solution trainning error
			#t = time.time()
			z = mappingToHighDimension(x_train, degree)
			theta_iterative = iterativeSol(z, y_train)
			#print "step01"
			yhat_train = mvarPredict(z, theta_iterative)
			#print "step02"
			mvarIteMse_train[i][num] = computeMSE(y_train, yhat_train)
			b += time.time()-t
			#print "step1"

			#for testing error
			z = mappingToHighDimension(x_test, degree)
			#solve using explicit solution
			#get mvar explicit solution testing error
			yhat = mvarPredict(z, theta_explicit)
			mvarLMse_test[i][num] = computeMSE(y_test, yhat)
			#print "step2"

			#solve using iterative solution
			#get mvar iterative solution testing error
			yhat = mvarPredict(z, theta_iterative)
			mvarIteMse_test[i][num] = computeMSE(y_test, yhat)
			#print "step3"


			#solve dual problem
			#get mvar dual problem solution trainning error
			t = time.time()
			yhat_train = solveDual(x_train, x_train, y_train)
			c += time.time()-t
			mvarDualMse_train[i][num] = computeMSE(y_train, yhat_train)
			#get mvar dual problem solution testing error
			yhat = solveDual(x_train, x_test, y_train)
			mvarDualMse_test[i][num] = computeMSE(y_test, yhat)
			
			#print "step4"


		#compute mean error
		for j in range(10):
			mvarL_trainError[i] += (mvarLMse_train[i][j] /10)
			mvarL_testError[i] += (mvarLMse_test[i][j] /10)
			mvarIte_trainError[i] += (mvarIteMse_train[i][j] /10)
			mvarIte_testError[i] += (mvarIteMse_test[i][j] /10)
			mvarDual_trainError[i] += (mvarDualMse_train[i][j] /10)
			mvarDual_testError[i] += (mvarDualMse_test[i][j] /10)
		print a/10
		print b/10
		print c/10
			

	print "multi variable explicit solution training error:\n",mvarL_trainError 
	print "multi variable explicit solution testing error:\n", mvarL_testError 
	print "multi variable iterative solution training error:\n",mvarIte_trainError 
	print "multi variable iterative solution testing error:\n",mvarIte_testError 
	print "multi variable dual problem solution training error:\n",mvarDual_trainError 
	print "multi variable dual problem solution testing error:\n",mvarDual_testError 
	

	
	

