import numpy as np
import random 
def dataGeneratorA():
	#randomly generate two clusters of data
	x_0 = np.zeros((100,2))
	x_1 = np.zeros((100,2))
	for i in range(100):
		x_0[i][0] = random.randint(0, 30)
		x_0[i][1] = random.randint(0, 30)
		x_1[i][0] = random.randint(35, 65)
		x_1[i][1] = random.randint(35, 65)
	#plt.plot(x_0[:,0], x_0[:,1], 'b.')
	#plt.plot(x_1[:,0], x_1[:,1], 'g.')
	#plt.show()

	#combine two clusters as feature matrix
	x = np.concatenate((x_0, x_1), axis = 0)
	
	#generate label vector
	y = np.zeros((200,1))
	for i in range(100):
		y[i][0] = -1
	for i in range(100,200):
		y[i][0] = 1

	data = np.concatenate((x, y), axis = 1)
	
	f=open("separable.txt",'w')
	for i in range(200):
		for j in range(3):
			f.write(str(data[i,j]))
			f.write("\t")
		f.write("\n")
	f.close()
	

def dataGeneratorB():
	x_0 = np.zeros((100,2))
	x_1 = np.zeros((100,2))
	for i in range(100):
		x_0[i][0] = random.randint(0, 30)
		x_0[i][1] = random.randint(0, 30)
		x_1[i][0] = random.randint(15, 45)
		x_1[i][1] = random.randint(15, 45)
	#plt.plot(x_0[:,0], x_0[:,1], 'b.')
	#plt.plot(x_1[:,0], x_1[:,1], 'g.')
	#plt.show()

	#combine two clusters as feature matrix
	x = np.concatenate((x_0, x_1), axis = 0)
	
	#generate label vector
	y = np.zeros((200, 1))
	for i in range(100):
		y[i][0] = -1
	for i in range(100,200):
		y[i][0] = 1
	
	data = np.concatenate((x, y), axis = 1)
	
	f=open("nonseparable.txt",'w')
	for i in range(200):
		for j in range(3):
			f.write(str(data[i,j]))
			f.write("\t")
		f.write("\n")
	f.close()

if __name__=="__main__":
	dataGeneratorA()
	dataGeneratorB()


