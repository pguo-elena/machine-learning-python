import numpy as np

from agents import Agent_single_sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC


def simulate_agents(agents, value, X, y, price_trials = 10):
	
	agent_wealths = {}
	
	for agent in agents:
		agent_wealths[agent] = 0
	
	num_products = X.shape[0]
	
	for p in range(num_products):        
		
		# Excellent or not?
		excellent = (y[p] == 'Excellent')
		
		for agent in agents:
			prob = agent.predict_prob_of_excellent(X[p])
			# try a range of prices            
			for pt in range(price_trials):                            
				price = ((2*pt+1)*value)/(2*price_trials)                                
				if agent.will_buy(value, price, prob):
					agent_wealths[agent] -= price
					if excellent:
						agent_wealths[agent] += value
	return agent_wealths


# Log-loss
def simulate_agents_logloss(agents, X, y):
	agent_ll = {}

	for agent in agents:
		agent_ll[agent] = 0

	num_products = X.shape[0]
	epsilon = 1e-10
	
	for agent in agents:
		for p in range(num_products):
			prob = agent.predict_prob_of_excellent(X[p])
			if y[p] == 'Excellent':
				agent_ll[agent] += -(np.log(prob+epsilon))
			else:
				agent_ll[agent] += -(np.log(1-prob+epsilon))
	return agent_ll

			

# 0/1 Loss
def simulate_agents_01_loss(agents, X, y):
	agent_error = {}

	for agent in agents:
		agent_error[agent] = 0
		
	num_products = X.shape[0]
	for agent in agents:	
		for p in range(num_products):
			prob = agent.predict_prob_of_excellent(X[p])
			if y[p] == 'Excellent':
				if prob < 0.5:
					agent_error[agent] += 1
			else:
				if prob >= 0.5:
					agent_error[agent] += 1
	return agent_error
			


if __name__ == '__main__':

	# You might need to change this depending on where the data is located
	data_path = "./"
	data_groups = ["dataset1"]
	for data_group in data_groups:
		
		train_file = data_path + data_group +  "_train.csv"
		val_file = data_path + data_group +  "_val.csv"
		test_file = data_path + data_group +  "_test.csv"
		
		train_data=np.loadtxt(train_file, dtype=str, delimiter=',', skiprows=1)
		X_train=train_data[:, 0:-1]=='T'
		y_train=train_data[:, -1]

		val_data=np.loadtxt(val_file, dtype=str, delimiter=',', skiprows=1)
		X_val=val_data[:, 0:-1]=='T'
		y_val=val_data[:, -1]

		test_data=np.loadtxt(test_file, dtype=str, delimiter=',', skiprows=1)
		X_test=test_data[:, 0:-1]=='T'
		y_test=test_data[:, -1]
		
		agents = []
		
		agents.append(Agent_single_sklearn("bnb", BernoulliNB()))
		agents.append(Agent_single_sklearn("lr", LogisticRegression()))
		agents.append(Agent_single_sklearn("svc",SVC(kernel = 'poly', degree = 4, probability = True, random_state = 0)))
		#agents.append(Agent_pguo3("pguo3"))

		# Add two more Agent_single_sklearn agents
		# One that uses LogisticRegression using default constructor
		#agents.append(Agent_single_sklearn("lr", change_this))
		# One that uses SVC with polynomial kernel degree of 4 and
		# probability estimates are turned on
		#agents.append(Agent_single_sklearn("svc", change_this))
		# Add your own agent; change mbilgic to your own hawk id
		#agents.append(Agent_mbilgic("mbilgic"))
		
		# Train the agents
		for agent in agents:
			agent.train(X_train, y_train, X_val, y_val)
		
		# Simulate the agents on test
		value = 1000
		agent_wealths = simulate_agents(agents, value, X_test, y_test)
		agent_logloss = simulate_agents_logloss(agents, X_test, y_test)
		agent_01_loss = simulate_agents_01_loss(agents, X_test, y_test)

		print agent_wealths
		print agent_logloss
		print agent_01_loss
		
		'''
		print "-" * 50
		print "SIMULATION RESULTS ON %s" %(data_group)
		print "-" * 50

		print "\nWealth (the larger the better)\n"
		for agent in agents:
			print "{}:\t\t${:,.2f}".format(agent, agent_wealths[agent])

		# Log-loss
		print "\nLog-loss (the smaller the better)\n"
		for agent in agents:
			print "{}:\t\t{:,.2f}".format(agent, agent_logloss[agent])
		
		'''
		'''
		epsilon = 1e-10
		for agent in agents:
			ll = 0
			num_products = X_test.shape[0]
			for p in range(num_products):
				prob = agent.predict_prob_of_excellent(X_test[p])
				if y_test[p] == 'Excellent':
					ll += -(np.log(prob+epsilon))
				else:
					ll += -(np.log(1-prob+epsilon))
			print "{}:\t\t{:,.2f}".format(agent, ll)
		'''
		'''
		# 0/1 Loss
		print "\n0/1 Loss (the smaller the better)\n"
		for agent in agents:
			print "{}:\t\t{:,.2f}".format(agent, agent_01_loss[agent])

		'''
		'''
		for agent in agents:
			error = 0
			num_products = X_test.shape[0]
			for p in range(num_products):
				prob = agent.predict_prob_of_excellent(X_test[p])
				if y_test[p] == 'Excellent':
					if prob < 0.5:
						error += 1
				else:
					if prob >= 0.5:
						error += 1
			print "{}:\t\t{:,d}".format(agent, error)
		'''
