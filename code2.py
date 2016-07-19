from __future__ import division
import numpy as np
import math
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

def oneDGaussian(x_train, y_train):
	m = np.ones(2)
	mu = np.zeros(2)
	sigma = np.zeros(2)
	alpha = np.ones(2)

	#compute mu
	for i in range(y_train.shape[0]):
		if y_train[i] == 1:
			m[0] += 1
			mu[0] += x_train[i]
		else:
			m[1] += 1
			mu[1] += x_train[i]
	#print"m", m
	mu[0] = mu[0]/m[0]
	mu[1] = mu[1]/m[1]

	#compute sigma
	for i in range(y_train.shape[0]):
		if y_train[i] == 1:
			sigma[0] += (x_train[i] - mu[0])**2
		else:
			sigma[1] += (x_train[i] - mu[1])**2
	sigma[0] = sigma[0]/m[0]
	sigma[1] = sigma[1]/m[1]

	#compute prior
	alpha[0] = m[0]/(m[0]+m[1])
	alpha[1] = m[1]/(m[0]+m[1])

	return mu, sigma, alpha

def oneDGaussianDiscriminantFun(x_test, mu, sigma, alpha):
	g = np.zeros((x_test.shape[0], 2))
	yhat = np.zeros(x_test.shape[0])
	#compute membership function
	for i in range(x_test.shape[0]):
		g[i][0] = -(x_test[i]-mu[0])**2
		g[i][1] = -(x_test[i]-mu[1])**2
		#g[i][0] = -np.log(sigma[0]) - ((x_test[i]-mu[0])**2)/(2*(sigma[0]**2))
		#g[i][1] = -np.log(sigma[1]) - ((x_test[i]-mu[1])**2)/(2*(sigma[1]**2))
		
		#g[i][0] = -np.log(sigma[0]) - ((x_test[i]-mu[0])**2)/(2*(sigma[0]**2)) + np.log(alpha[0])
		#g[i][1] = -np.log(sigma[1]) - ((x_test[i]-mu[1])**2)/(2*(sigma[1]**2)) + np.log(alpha[1])
		if g[i][0] > g[i][1]:
			yhat[i] = 1
		else:
			yhat[i] = 2
	return yhat


def prob1():
	fname = "Skin_Nonskin.txt"
	datasets = np.loadtxt(fname)
	x = datasets[:, 0]
	y = datasets[:, -1]
	kf = KFold(x.shape[0], n_folds = 3, shuffle = True, random_state = 0)
	
	#count = 0
	tp = 1
	fp = 1
	fn = 1
	tn = 1
	for train_index, test_index in kf:
		#count += 1
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		mu, sigma, alpha = oneDGaussian(x_train, y_train)
		#print "mu", mu 
		#print "sigma", sigma
		yhat = oneDGaussianDiscriminantFun(x_test, mu, sigma, alpha)
		precision = 0
		recall = 0
		f_measure = 0
		accuracy = 0
	
		for i in range(y_test.shape[0]):
			if y_test[i] == 1:
				if yhat[i] == 1:
					tp += 1
				else:
					fn += 1
			else:
				if yhat[i] == 1:
					fp += 1
				else:
					tn += 1 
	#confusion matrix
	c = np.zeros((2,2))
	c[0][0] = tp
	c[0][1] = fp
	c[1][0] = fn
	c[1][1] = tn
	#compute precision
	precision = tp/(tp+fp)
	#compute recall
	recall = tp/(tp+fn)
	#compute f-measure
	f_measure = 2*((precision*recall)/(precision+recall))
	#compute accuracy
	accuracy = (tp+tn)/(tp+tn+fp+fn)
	print "-" * 50
	print "problem1"
	print "-" * 50
	print "confusion matrix", c
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy
	print "-"*50
	print "\n"

def nDGaussian(x_train, y_train):
	m = np.ones(2)
	mu = np.zeros((2, x_train.shape[1]))
	sigma = np.zeros((2, x_train.shape[1], x_train.shape[1]))
	alpha = np.zeros(2)

	#compute mu
	for i in range(x_train.shape[0]):
		if y_train[i] == 1:
			m[0] += 1
			mu[0] += x_train[i]
		else:
			m[1] += 1
			mu[1] += x_train[i]
		#print"m", m
	mu[0] = mu[0]/m[0]
	mu[1] = mu[1]/m[1]

	#compute sigma
	for i in range(x_train.shape[0]):
		if y_train[i] == 1:
			diff = x_train[i] - mu[0]
			diff = np.reshape(diff, (diff.shape[0], 1))
			diff_t = np.reshape(diff, (1, diff.shape[0]))
			sigma[0] += np.dot(diff, diff_t)
		else:
			diff = x_train[i] - mu[1]
			diff = np.reshape(diff, (diff.shape[0], 1))
			diff_t = np.reshape(diff, (1, diff.shape[0]))
			sigma[1] += np.dot(diff, diff_t)

	sigma[0] = sigma[0]/m[0]
	sigma[1] = sigma[1]/m[1]

	#compute prior
	alpha[0] = m[0]/(m[0]+m[1])
	alpha[1] = m[1]/(m[0]+m[1])
	return mu, sigma, alpha

def nDGaussianDiscriminantFun(x_test, mu, sigma, alpha):
	g = np.zeros((x_test.shape[0], 2))
	yhat = np.zeros(x_test.shape[0])
	
	#compute membership function
	for i in range(x_test.shape[0]):
		diff_0 = x_test[i] - mu[0]
		diff_0 = np.reshape(diff_0,(diff_0.shape[0], 1))
		diff_t0 = np.reshape(diff_0, (1, diff_0.shape[0]))
		g[i][0] = - np.dot(np.dot(diff_t0, np.linalg.inv(sigma[0])), diff_0)/2

		diff_1 = x_test[i] - mu[1]
		diff_1 = np.reshape(diff_1,(diff_1.shape[0], 1))
		diff_t1 = np.reshape(diff_1, (1, diff_1.shape[0]))
		g[i][1] = - np.dot(np.dot(diff_t1, np.linalg.inv(sigma[1])), diff_1)/2

		#compute discriminant function
		if g[i][0] > g[i][1]:
			yhat[i] = 1
		else:
			yhat[i] = 2
	return yhat


def prob2():
	fname = "Skin_Nonskin.txt"
	datasets = np.loadtxt(fname)
	x = datasets[:, :-1]
	y = datasets[:, -1]
	kf = KFold(x.shape[0], n_folds = 3, shuffle = True, random_state = 0)
	tp = 1
	fp = 1
	fn = 1
	tn = 1
	precision = 0
	recall = 0
	f_measure = 0
	accuracy = 0
	x_plot = []
	y_plot = []
	for train_index, test_index in kf:
		#count += 1
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		mu, sigma, alpha = nDGaussian(x_train, y_train)
		#print "mu", mu 
		#print "sigma", sigma
		tpp=0
		fnn=0
		tnn=0
		yhat = nDGaussianDiscriminantFun(x_test, mu, sigma, alpha)
	
		for i in range(y_test.shape[0]):
			if y_test[i] == 1:
				if yhat[i] == 1:
					tp += 1
					tpp+=1
				else:
					fn += 1
					fnn +=1
			else:
				if yhat[i] == 1:
					fp += 1
				else:
					tn += 1 
					tnn+=1
		precision = tpp/(tpp+fnn)
		recall = tpp/(tpp+fnn)
		x_plot.append(precision)
		y_plot.append(recall)

	plt.plot(x_plot, y_plot)
	plt.xlabel('precision')
	plt.ylabel('recall')
	plt.show()
	#confusion matrix
	c = np.zeros((2,2))
	c[0][0] = tp
	c[0][1] = fp
	c[1][0] = fn
	c[1][1] = tn	
	#compute precision
	precision = tp/(tp+fp)
	#compute recall
	recall = tp/(tp+fn)
	#compute f-measure
	f_measure = 2*((precision*recall)/(precision+recall))
	#compute accuracy
	accuracy = (tp+tn)/(tp+tn+fp+fn)
	print "-" * 50
	print "problem2"
	print "-" * 50
	print "confusion matrix", c
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy
	print "-"*50
	print "\n"


def prob3():
	fname = "wine.data"
	datasets = np.loadtxt(fname, delimiter = ',')
	x = datasets[: , 1:]
	y = datasets[:, 0]

	kf = KFold(x.shape[0], n_folds = 3, shuffle = True, random_state = 0)

	confusion = np.ones((3, 3))
	precision = np.ones((3, 1))
	recall = np.ones((3, 1))
	f_measure = np.ones((3, 1))
	accuracy = 1
	
	for train_index, test_index in kf:
		#count += 1
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		'''
		---------------start estimate the model parameters
		--------------------------------------------------
		'''
		m = np.ones(3)
		mu = np.zeros((3, x_train.shape[1]))
		sigma = np.zeros((3, x_train.shape[1], x_train.shape[1]))
		alpha = np.zeros(3)
	
		#compute mu
		for i in range(x_train.shape[0]):
			if y_train[i] == 1:
				m[0] += 1
				mu[0] += x_train[i]
			elif y_train[i] == 2:
				m[1] += 1
				mu[1] += x_train[i]
			else:
				m[2] += 1
				mu[2] += x_train[i]
			#print"m", m
		mu[0] = mu[0]/m[0]
		mu[1] = mu[1]/m[1]
		mu[2] = mu[2]/m[2]
	
		#compute sigma
		for i in range(x_train.shape[0]):
			if y_train[i] == 1:
				diff = x_train[i] - mu[0]
				diff = np.reshape(diff, (diff.shape[0], 1))
				diff_t = np.reshape(diff, (1, diff.shape[0]))
				sigma[0] += np.dot(diff, diff_t)
			elif y_train[i] == 2:
				diff = x_train[i] - mu[1]
				diff = np.reshape(diff, (diff.shape[0], 1))
				diff_t = np.reshape(diff, (1, diff.shape[0]))
				sigma[1] += np.dot(diff, diff_t)
			else:
				diff = x_train[i] - mu[2]
				diff = np.reshape(diff, (diff.shape[0], 1))
				diff_t = np.reshape(diff, (1, diff.shape[0]))
				sigma[2] += np.dot(diff, diff_t)
	
		sigma[0] = sigma[0]/m[0]
		sigma[1] = sigma[1]/m[1]
		sigma[2] = sigma[2]/m[2]
	
		#compute prior
		alpha[0] = m[0]/(m[0]+m[1]+m[2])
		alpha[1] = m[1]/(m[0]+m[1]+m[2])
		alpha[2] = m[2]/(m[0]+m[1]+m[2])
		'''
		---------------end estimate the model parameters
		------------------------------------------------
		'''	

		'''
		--------------start compute discriminant function
		-------------------------------------------------
		'''
		g = np.zeros((x_test.shape[0], 3))
		yhat = np.zeros(x_test.shape[0])
		
		#compute membership function
		for i in range(x_test.shape[0]):
			diff_0 = x_test[i] - mu[0]
			diff_0 = np.reshape(diff_0,(diff_0.shape[0], 1))
			diff_t0 = np.reshape(diff_0, (1, diff_0.shape[0]))
			g[i][0] = - np.dot(np.dot(diff_t0, np.linalg.inv(sigma[0])), diff_0)/2
	
			diff_1 = x_test[i] - mu[1]
			diff_1 = np.reshape(diff_1,(diff_1.shape[0], 1))
			diff_t1 = np.reshape(diff_1, (1, diff_1.shape[0]))
			g[i][1] = - np.dot(np.dot(diff_t1, np.linalg.inv(sigma[1])), diff_1)/2

			diff_2 = x_test[i] - mu[2]
			diff_2 = np.reshape(diff_2,(diff_2.shape[0], 1))
			diff_t2 = np.reshape(diff_2, (1, diff_2.shape[0]))
			g[i][2] = - np.dot(np.dot(diff_t2, np.linalg.inv(sigma[2])), diff_2)/2
	
			#compute discriminant function
			if g[i][0] > g[i][1]:
				if g[i][1] > g[i][2]:
					yhat[i] = 1
				else:
					if g[i][2] > g[i][0]:
						yhat[i] = 3
					else:
						yhat[i] = 1
			else:
				if g[i][0] > g[i][2]:
					yhat[i] = 2
				else:
					if g[i][1] > g[i][2]:
						yhat[i] = 2
					else:
						yhat[i] = 3				
		'''
		--------------end compute discriminant function
		------------------------------------------------
		'''

		'''
		--------------evaluate performance
		----------------------------------
		'''

		#compute confusion matrix
		for i in range(yhat.shape[0]):
			a = yhat[i]
			b = y_test[i]
			confusion[a-1][b-1] += 1

	#compute precision
	nume = 0
	for i in range(3):
		nume = confusion[i][i]
		deno = 0
		for j in range(3):
			deno += confusion[i][j]
		precision[i] = nume/deno

	#compute recall
	for j in range(3):
		nume = confusion[j][j]
		deno = 0
		for i in range(3):
			deno += confusion[i][j]
		recall[j] = nume/deno
	
	#compute f-measure
	for i in range(3):
		f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
	
	#compute accuracy
	nume = 0
	deno = 0
	for i in range(3):
		for j in range(3):
			if i == j:
				nume += confusion[i][j]
			deno += confusion[i][j]
	accuracy = nume/deno

	#print result
	print "-" * 50
	print "problem 3"
	print "-" * 50
	print "confusion matrix", confusion
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy
	print "-"*50
	print "\n"
			

def prob4():
	fname = "SPECT.train"
	datasets = np.loadtxt(fname, delimiter = ',')
	x = datasets[: , 1:]
	y = datasets[:, 0]
	kf = KFold(x.shape[0], n_folds = 3, shuffle = True, random_state = 0)

	tp = 1
	fp = 1
	fn = 1
	tn = 1
	precision = 0
	recall = 0
	f_measure = 0
	accuracy = 0	

	for train_index, test_index in kf:
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		alpha = np.ones((2, x_train.shape[1]))
		m = np.ones((3))
		g = np.zeros((2, x_test.shape[0]))
		yhat = np.zeros(x_test.shape[0])
		
		#-------------------------estimate parameters
		
		for i in range(x_train.shape[1]):
			for j in range(x_train.shape[0]):
				if y_train[j] == 0:
					alpha[0][i] += x_train[j][i]
					m[0] += 1
					m[2] += 1
				else:
					alpha[1][i] += x_train[j][i]
					m[1] += 1
					m[2] += 1
		alpha[0] = alpha[0]/m[0]
		alpha[1] = alpha[1]/m[1]
		p_y0 = m[0]/m[2]
		p_y1 = m[1]/m[2]

		#---------------------------------------------

		#----------------compute descriminant function

		for j in range(x_test.shape[0]):
			for i in range(x_test.shape[1]):
				g[0][j] += (x_test[j][i]*np.log(alpha[0][i]) + (1-x_test[j][i])*np.log(1-alpha[0][i]) + np.log(p_y0))
				#last term could set to 1/2 for equal prior
				#g[0][j] += (x_test[j][i]*np.log(alpha[0][i]) + (1-x_test[j][i])*np.log(1-alpha[0][i]) + np.log(1/2))
				g[1][j] += (x_test[j][i]*np.log(alpha[1][i]) + (1-x_test[j][i])*np.log(1-alpha[1][i]) + np.log(p_y1))
				#g[1][j] += (x_test[j][i]*np.log(alpha[1][i]) + (1-x_test[j][i])*np.log(1-alpha[1][i]) + np.log(1/2))
			if g[0][j] > g[1][j]:
				yhat[j] = 0
			else:
				yhat[j] = 1
		#-----------------------------------------------

		#---------------------------evaluate performance

		for i in range(y_test.shape[0]):
			if y_test[i] == 0:
				if yhat[i] == 0:
					tp += 1
				else:
					fn += 1
			else:
				if yhat[i] == 0:
					fp += 1
				else:
					tn += 1 
	#confusion matrix
	c = np.zeros((2,2))
	c[0][0] = tp
	c[0][1] = fp
	c[1][0] = fn
	c[1][1] = tn	
	#compute precision
	precision = tp/(tp+fp)
	#compute recall
	recall = tp/(tp+fn)
	#compute f-measure
	f_measure = 2*((precision*recall)/(precision+recall))
	#compute accuracy
	accuracy = (tp+tn)/(tp+tn+fp+fn)

	print "-" * 50
	print "problem 4"
	print "-" * 50
	print "confusion matrix", c
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy
	print "-"*50
	print "\n"


def prob5():
	fname = "monks.train"
	datasets = np.loadtxt(fname)
	x = datasets[:, 1:]
	y = datasets[:, 0]
	kf = KFold(x.shape[0], n_folds = 15, shuffle = True, random_state = 0)

	tp = 1
	fp = 1
	fn = 1
	tn = 1
	precision = 0
	recall = 0
	f_measure = 0
	accuracy = 0	

	for train_index, test_index in kf:
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		
		alpha = np.ones((2, x_train.shape[1]))
		m = np.ones((3))
		g = np.zeros((2, x_test.shape[0]))
		yhat = np.zeros(x_test.shape[0])
		p = np.zeros(x_test.shape[0])
		
		#-------------------------estimate parameters
		
		for i in range(x_train.shape[1]):
			for j in range(x_train.shape[0]):
				if y_train[j] == 0:
					alpha[0][i] += x_train[j][i]
					m[0] += 1
					m[2] += 1
				else:
					alpha[1][i] += x_train[j][i]
					m[1] += 1
					m[2] += 1
		
		alpha[0] = alpha[0]/m[0]
		alpha[1] = alpha[1]/m[1]
		p_y0 = m[0]/m[2]
		p_y1 = m[1]/m[2]

		#---------------------------------------------

		#----------------compute descriminant function
		for j in range(x_test.shape[0]):
			for i in range(x_test.shape[1]):
				p[j] += x_test[j][i]
		
		for j in range(x_test.shape[0]):
			for i in range(x_test.shape[1]):
				a = np.log(math.factorial(p[j])/(math.factorial(x[j][i])*math.factorial(p[j]-x[j][i])))
				b = x[j][i]*np.log(alpha[0][i])
				c = (p[j]-x[j][i])*np.log(1-alpha[0][i])
				g[0][j] += (a+b+c+np.log(p_y0)) 
				#g[0][j] += (a+b+c+np.log(1/2))
				
				e = x[j][i]*np.log(alpha[1][i])
				f = (p[j]-x[j][i])*np.log(1-alpha[1][i])
				g[1][j] += (a+e+f+np.log(p_y1)) 
				#g[1][j] += (a+e+f+np.log(1/2))
			
			if g[0][j] > g[1][j]:
				yhat[j] = 0
			else:
				yhat[j] = 1
		#-----------------------------------------------

		#---------------------------evaluate performance

		for i in range(y_test.shape[0]):
			if y_test[i] == 0:
				if yhat[i] == 0:
					tp += 1
				else:
					fn += 1
			else:
				if yhat[i] == 0:
					fp += 1
				else:
					tn += 1 
	#confusion matrix
	c = np.zeros((2,2))
	c[0][0] = tp
	c[0][1] = fp
	c[1][0] = fn
	c[1][1] = tn	
	#compute precision
	precision = tp/(tp+fp)
	#compute recall
	recall = tp/(tp+fn)
	#compute f-measure
	f_measure = 2*((precision*recall)/(precision+recall))
	#compute accuracy
	accuracy = (tp+tn)/(tp+tn+fp+fn)

	print "-" * 50
	print "problem 5"
	print "-" * 50
	print "confusion matrix", c
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy
	print "-"*50
	print "\n"
	
	

if __name__=="__main__":
	#prob1()
	prob2()
	#prob3()
	#prob4()
	#prob5()

