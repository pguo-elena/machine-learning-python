from __future__ import division
import numpy as np
import math
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
from scipy.spatial.distance import euclidean


def hardMarginSVM(x, y):
	row, colum = x.shape
	w = np.zeros((1, colum))
	q = -np.ones((row,1))
	G = -np.identity(row)
	h = np.zeros((row, 1))
	b = np.zeros((1,1))

	y_t = np.transpose(y)
	x_t = np.transpose(x)
	
	p = np.dot(y,y_t)*np.dot(x,x_t)
	A = y_t
	solvers.options['show_progress'] = False
	sol = solvers.qp(matrix(p),matrix(q),matrix(G),matrix(h),matrix(A),matrix(b))
	alpha = np.array(sol['x'])
	#print alpha

	#figure out support vector and plot data(that are not sv)
	sv_index = list()
	for i in range(row):
		if alpha[i][0] > 1e-4:
			sv_index.append(i)
		else:
			if y[i][0] == -1:
				plt.plot(x[i][0], x[i][1], 'b.')
			elif y[i][0] == 1:
				plt.plot(x[i][0], x[i][1], 'g.')

	#plot sv
	for i in sv_index:
		if y[i][0] == -1:
			plt.plot(x[i][0], x[i][1], 'bo')
		elif y[i][0] == 1:
			plt.plot(x[i][0], x[i][1], 'go')
	plt.show()
	
	#compute w
	for i in sv_index:
		for j in range(colum):
			w[0][j] += alpha[i][0]*y[i]*x[i][j]
	
	#compute w0
	num = 0
	w0 = 0
	wx = 0
	for i in sv_index:
		num += 1
		for j in range(colum):
			wx += w[0][j]*x[i][j]
		w0 += y[i] - wx
	w0 = w0/num

	#print w
	return w, w0,alpha, sv_index

def predict(w, w0, x, y, cmat):
	row, colum = x.shape
	yhat = np.zeros(row)
	w_t = np.transpose(w)
	for i in range(row):
		yhat[i] =  np.dot(x[i], w_t) + w0
		#print yhat
		if yhat[i] < 0:
			if y[i] == -1:
				cmat[0][0] += 1
			elif y[i] == 1:
				cmat[0][1] += 1
		elif yhat[i] > 0:
			if y[i] == -1:
				cmat[1][0] += 1
			elif y[i] == 1:
				cmat[1][1] += 1
	return cmat

def question2(x, y):
	#for separable dataset
	kf = KFold(x.shape[0], n_folds = 2, shuffle = True, random_state = 0)
	cmat = np.ones((2, 2))

	for train_index, test_index in kf:
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#training, get parameters
		w, w0, alpha, sv_index = hardMarginSVM(x_train, y_train)
		#print alpha.shape
		#print w.shape

		#test performance
		cmat = predict(w, w0, x_test, y_test, cmat)

	num_class = cmat.shape[0]
	precision = np.ones(num_class)
	recall = np.ones(num_class)
	f_measure = np.ones(num_class)
	accuracy = 0
	#compute precision
	nume = 0
	for i in range(cmat.shape[0]):
		nume = cmat[i][i]
		deno = 0
		for j in range(cmat.shape[1]):
			deno += cmat[i][j]
		precision[i] = nume/deno

	#compute recall
	for j in range(cmat.shape[1]):
		nume = cmat[j][j]
		deno = 0
		for i in range(cmat.shape[0]):
			deno += cmat[i][j]
		recall[j] = nume/deno
	
	#compute f-measure
	for i in range(cmat.shape[0]):
		f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
	
	#compute accuracy
	nume = 0
	deno = 0
	for i in range(cmat.shape[0]):
		for j in range(cmat.shape[1]):
			if i == j:
				nume += cmat[i][j]
			deno += cmat[i][j]
	accuracy = nume/deno
	print "confusion matrix:", cmat
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy
	
def softMarginSVM(x, y):
	c = 1
	row, colum = x.shape
	w = np.zeros((1, colum))
	q = -np.ones((row,1))
	upper_G = -np.identity(row)
	lower_G = np.identity(row)
	G = np.concatenate((upper_G, lower_G,), axis = 0)
	upper_h = np.zeros((row, 1))
	lower_h = np.ones((row, 1))*c
	h = np.concatenate((upper_h, lower_h), axis = 0)
	b = np.zeros((1,1))

	y_t = np.transpose(y)
	x_t = np.transpose(x)
	
	p = np.dot(y,y_t)*np.dot(x,x_t)
	A = y_t
	solvers.options['show_progress'] = False
	sol = solvers.qp(matrix(p),matrix(q),matrix(G),matrix(h),matrix(A),matrix(b))
	alpha = np.array(sol['x'])
	#print alpha

	#figure out support vector and plot data(that are not sv)
	sv_index = list()
	for i in range(row):
		if alpha[i][0] > 1e-4:
			sv_index.append(i)
		else:
			if y[i][0] == -1:
				plt.plot(x[i][0], x[i][1], 'b.')
			elif y[i][0] == 1:
				plt.plot(x[i][0], x[i][1], 'g.')

	#plot sv
	for i in sv_index:
		if y[i][0] == -1:
			plt.plot(x[i][0], x[i][1], 'bo')
		elif y[i][0] == 1:
			plt.plot(x[i][0], x[i][1], 'go')
	plt.show()
	
	#compute w
	for i in sv_index:
		for j in range(colum):
			w[0][j] += alpha[i][0]*y[i]*x[i][j]
	
	#compute w0
	num = 0
	w0 = 0
	wx = 0
	for i in sv_index:
		num += 1
		for j in range(colum):
			wx += w[0][j]*x[i][j]
		w0 += y[i] - wx
	w0 = w0/num

	#print w
	return w, w0, alpha, sv_index

def question4(x, y):
	#for separable dataset
	kf = KFold(x.shape[0], n_folds = 2, shuffle = True, random_state = 0)
	cmat = np.ones((2, 2))

	for train_index, test_index in kf:
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#training, get parameters
		w, w0, alpha, sv_index = softMarginSVM(x_train, y_train)
		#print alpha.shape
		#print w.shape

		#test performance
		cmat = predict(w, w0, x_test, y_test, cmat)

	num_class = cmat.shape[0]
	precision = np.ones(num_class)
	recall = np.ones(num_class)
	f_measure = np.ones(num_class)
	accuracy = 0
	#compute precision
	nume = 0
	for i in range(cmat.shape[0]):
		nume = cmat[i][i]
		deno = 0
		for j in range(cmat.shape[1]):
			deno += cmat[i][j]
		precision[i] = nume/deno

	#compute recall
	for j in range(cmat.shape[1]):
		nume = cmat[j][j]
		deno = 0
		for i in range(cmat.shape[0]):
			deno += cmat[i][j]
		recall[j] = nume/deno
	
	#compute f-measure
	for i in range(cmat.shape[0]):
		f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
	
	#compute accuracy
	nume = 0
	deno = 0
	for i in range(cmat.shape[0]):
		for j in range(cmat.shape[1]):
			if i == j:
				nume += cmat[i][j]
			deno += cmat[i][j]
	accuracy = nume/deno
	print "confusion matrix:", cmat
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy

def obtainSV(x, y, alpha, sv_index):
	num = len(sv_index)
	sv_alpha = list()
	sv_x = list()
	sv_y = list()

	for i in sv_index:
		sv_alpha.append(alpha[i])
		sv_x.append(x[i])
		sv_y.append(y[i])

	sv_alpha = np.array(sv_alpha)
	sv_x = np.array(sv_x)
	sv_y = np.array(sv_y)

	return sv_alpha, sv_x, sv_y

def polyKernel(w0, sv_alpha, sv_x, sv_y, x, y, cmat_poly):
	q = 2
	row, colum = x.shape
	num = sv_alpha.shape[0]
	yhat = np.zeros(row)
	
	for i in range(row):
		x_t = np.transpose(x[i])
		for j in range(num):
			yhat[i] += sv_alpha[j]*sv_y[j]*(np.dot(sv_x[j],x_t)**q)
		yhat[i] = yhat[i] + w0
		if yhat[i] < 0:
			if y[i] == -1:
				cmat_poly[0][0] += 1
			elif y[i] == 1:
				cmat_poly[0][1] += 1
		elif yhat[i] > 0:
			if y[i] == -1:
				cmat_poly[1][0] += 1
			elif y[i] == 1:
				cmat_poly[1][1] += 1
	return cmat_poly

def gaussianKernel(w0, sv_alpha, sv_x, sv_y, x, y, cmat_gaussian):
	row, colum = x.shape
	num = sv_alpha.shape[0]
	yhat = np.zeros(row)
	sigma = 0.1
	
	for i in range(row):
		#x_t = np.transpose(x[i])
		for j in range(num):
			yhat[i] += sv_alpha[j]*sv_y[j]*(np.exp(euclidean(sv_x[j], x[i])/((-2)*(sigma**2))))
		if yhat[i] < 0:
			if y[i] == -1:
				cmat_gaussian[0][0] += 1
			elif y[i] == 1:
				cmat_gaussian[0][1] += 1
		elif yhat[i] > 0:
			if y[i] == -1:
				cmat_gaussian[1][0] += 1
			elif y[i] == 1:
				cmat_gaussian[1][1] += 1
	return cmat_gaussian

def question5(x, y):
	#for separable dataset
	kf = KFold(x.shape[0], n_folds = 2, shuffle = True, random_state = 0)
	cmat_poly = np.ones((2, 2))
	cmat_gaussian = np.ones((2, 2))

	for train_index, test_index in kf:
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#training, get parameters
		w, w0, alpha, sv_index = softMarginSVM(x_train, y_train)
		sv_alpha, sv_x, sv_y = obtainSV(x_train, y_train, alpha, sv_index)
		
		#use kernel function, test performance
		cmat_poly = polyKernel(w0, sv_alpha, sv_x, sv_y, x_test, y_test, cmat_poly)
		cmat_gaussian = gaussianKernel(w0, sv_alpha, sv_x, sv_y, x_test, y_test, cmat_gaussian)
		
	#performance of polynomial kernel function	
	num_class = cmat_poly.shape[0]
	precision = np.ones(num_class)
	recall = np.ones(num_class)
	f_measure = np.ones(num_class)
	accuracy = 0
	#compute precision
	nume = 0
	for i in range(cmat_poly.shape[0]):
		nume = cmat_poly[i][i]
		deno = 0
		for j in range(cmat_poly.shape[1]):
			deno += cmat_poly[i][j]
		precision[i] = nume/deno

	#compute recall
	for j in range(cmat_poly.shape[1]):
		nume = cmat_poly[j][j]
		deno = 0
		for i in range(cmat_poly.shape[0]):
			deno += cmat_poly[i][j]
		recall[j] = nume/deno
	
	#compute f-measure
	for i in range(cmat_poly.shape[0]):
		f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
	
	#compute accuracy
	nume = 0
	deno = 0
	for i in range(cmat_poly.shape[0]):
		for j in range(cmat_poly.shape[1]):
			if i == j:
				nume += cmat_poly[i][j]
			deno += cmat_poly[i][j]
	accuracy = nume/deno
	print "use polynomial kernel function"
	print "confusion matrix:", cmat_poly
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy

	#performance of gaussian kernel function
	num_class = cmat_gaussian.shape[0]
	precision = np.ones(num_class)
	recall = np.ones(num_class)
	f_measure = np.ones(num_class)
	accuracy = 0
	#compute precision
	nume = 0
	for i in range(cmat_gaussian.shape[0]):
		nume = cmat_gaussian[i][i]
		deno = 0
		for j in range(cmat_gaussian.shape[1]):
			deno += cmat_gaussian[i][j]
		precision[i] = nume/deno

	#compute recall
	for j in range(cmat_gaussian.shape[1]):
		nume = cmat_gaussian[j][j]
		deno = 0
		for i in range(cmat_gaussian.shape[0]):
			deno += cmat_gaussian[i][j]
		recall[j] = nume/deno
	
	#compute f-measure
	for i in range(cmat_gaussian.shape[0]):
		f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
	
	#compute accuracy
	nume = 0
	deno = 0
	for i in range(cmat_gaussian.shape[0]):
		for j in range(cmat_gaussian.shape[1]):
			if i == j:
				nume += cmat_gaussian[i][j]
			deno += cmat_gaussian[i][j]
	accuracy = nume/deno
	print "performance of gaussian kernel function"
	print "confusion matrix:", cmat_gaussian
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy

def softMarginSVM_ex(x, y):
	c = 1
	row, colum = x.shape
	w = np.zeros((1, colum))
	q = -np.ones((row,1))
	upper_G = -np.identity(row)
	lower_G = np.identity(row)
	G = np.concatenate((upper_G, lower_G,), axis = 0)
	upper_h = np.zeros((row, 1))
	lower_h = np.ones((row, 1))*c
	h = np.concatenate((upper_h, lower_h), axis = 0)
	b = np.zeros((1,1))

	y_t = np.transpose(y)
	x_t = np.transpose(x)
	
	p = np.dot(y,y_t)*np.dot(x,x_t)
	A = y_t
	solvers.options['show_progress'] = False
	sol = solvers.qp(matrix(p),matrix(q),matrix(G),matrix(h),matrix(A),matrix(b))
	alpha = np.array(sol['x'])
	#print alpha

	#figure out support vector and plot data(that are not sv)
	sv_index = list()
	for i in range(row):
		if alpha[i][0] > 1e-4:
			sv_index.append(i)
		
	#compute w
	for i in sv_index:
		for j in range(colum):
			w[0][j] += alpha[i][0]*y[i]*x[i][j]
	
	#compute w0
	num = 0
	w0 = 0
	wx = 0
	for i in sv_index:
		num += 1
		for j in range(colum):
			wx += w[0][j]*x[i][j]
		w0 += y[i] - wx
	w0 = w0/num

	#print w
	return w, w0, alpha, sv_index

def question5_exData(x, y):
	#for separable dataset
	kf = KFold(x.shape[0], n_folds = 2, shuffle = True, random_state = 0)
	cmat_poly = np.ones((2, 2))
	cmat_gaussian = np.ones((2, 2))

	for train_index, test_index in kf:
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#training, get parameters
		w, w0, alpha, sv_index = softMarginSVM_ex(x_train, y_train)
		sv_alpha, sv_x, sv_y = obtainSV(x_train, y_train, alpha, sv_index)
		
		#use kernel function, test performance
		cmat_poly = polyKernel(w0, sv_alpha, sv_x, sv_y, x_test, y_test, cmat_poly)
		cmat_gaussian = gaussianKernel(w0, sv_alpha, sv_x, sv_y, x_test, y_test, cmat_gaussian)
		
	#performance of polynomial kernel function	
	num_class = cmat_poly.shape[0]
	precision = np.ones(num_class)
	recall = np.ones(num_class)
	f_measure = np.ones(num_class)
	accuracy = 0
	#compute precision
	nume = 0
	for i in range(cmat_poly.shape[0]):
		nume = cmat_poly[i][i]
		deno = 0
		for j in range(cmat_poly.shape[1]):
			deno += cmat_poly[i][j]
		precision[i] = nume/deno

	#compute recall
	for j in range(cmat_poly.shape[1]):
		nume = cmat_poly[j][j]
		deno = 0
		for i in range(cmat_poly.shape[0]):
			deno += cmat_poly[i][j]
		recall[j] = nume/deno
	
	#compute f-measure
	for i in range(cmat_poly.shape[0]):
		f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
	
	#compute accuracy
	nume = 0
	deno = 0
	for i in range(cmat_poly.shape[0]):
		for j in range(cmat_poly.shape[1]):
			if i == j:
				nume += cmat_poly[i][j]
			deno += cmat_poly[i][j]
	accuracy = nume/deno
	print "use polynomial kernel function"
	print "confusion matrix:", cmat_poly
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy

	#performance of gaussian kernel function
	num_class = cmat_gaussian.shape[0]
	precision = np.ones(num_class)
	recall = np.ones(num_class)
	f_measure = np.ones(num_class)
	accuracy = 0
	#compute precision
	nume = 0
	for i in range(cmat_gaussian.shape[0]):
		nume = cmat_gaussian[i][i]
		deno = 0
		for j in range(cmat_gaussian.shape[1]):
			deno += cmat_gaussian[i][j]
		precision[i] = nume/deno

	#compute recall
	for j in range(cmat_gaussian.shape[1]):
		nume = cmat_gaussian[j][j]
		deno = 0
		for i in range(cmat_gaussian.shape[0]):
			deno += cmat_gaussian[i][j]
		recall[j] = nume/deno
	
	#compute f-measure
	for i in range(cmat_gaussian.shape[0]):
		f_measure[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
	
	#compute accuracy
	nume = 0
	deno = 0
	for i in range(cmat_gaussian.shape[0]):
		for j in range(cmat_gaussian.shape[1]):
			if i == j:
				nume += cmat_gaussian[i][j]
			deno += cmat_gaussian[i][j]
	accuracy = nume/deno
	print "performance of gaussian kernel function"
	print "confusion matrix:", cmat_gaussian
	print "precision:", precision
	print "recall:", recall
	print "f-measure:", f_measure
	print "accuracy:", accuracy

if __name__=="__main__":
	#load separable data
	f1 = "separable.txt"
	dataset = np.loadtxt(f1)
	x_sep = dataset[:,0:-1]
	y_sep = dataset[:, -1]
	y_sep = np.reshape(y_sep, (200, 1))


	#load non-separable data
	f2 = "nonseparable.txt"
	dataset = np.loadtxt(f2)
	x_nonsep = dataset[:,0:-1]
	y_nonsep = dataset[:, -1]
	y_nonsep = np.reshape(y_nonsep, (200, 1))

	#question2(x_sep, y_sep)
	#question2(x_nonsep, y_nonsep)

	#question4(x_sep, y_sep)
	#question4(x_nonsep, y_nonsep)

	#question5(x_sep, y_sep)
	#question5(x_nonsep, y_nonsep)
	f3 = "heart.dat"
	dataset = np.loadtxt(f3)
	x = dataset[:,0:-1]
	y = dataset[:, -1]
	y = np.reshape(y, (y.shape[0], 1))
	for i in range(y.shape[0]):
		if y[i] == 2:
			y[i] = -1
	
	#question5_exData(x, y)

	f4 = "monks.train"
	datasets = np.loadtxt(f4)
	x = datasets[:, 1:]
	y = datasets[:, 0]
	y = np.reshape(y, (y.shape[0], 1))
	for i in range(y.shape[0]):
		if y[i] == 0:
			y[i] = -1
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if x[i][j] == "null":
				x[i][j] = 0
	#question5_exData(x, y)

	#question 6
	dataset = np.loadtxt(f1)
	x1 = dataset[:100,0:-1]
	y1 = dataset[:100, -1]
	x2 = dataset[101:110, 0:-1]
	y2 = dataset[101:110, -1]
	x = np.concatenate((x1, x2), axis = 0)
	y = np.concatenate((y1, y2), axis = 0)
	y = np.reshape(y, (y.shape[0], 1))
	question2(x, y)

