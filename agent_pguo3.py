from __future__ import division
import numpy as np

from agents import Agent_single_sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from agents import Agent


class Agent_pguo3(Agent):
	def __init__(self,name):
		self.name = name

	#wealth
	def simulate_agents_wealths(self, agents, value, X, y, price_trials = 10):
	
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
	def simulate_agents_logloss(self, agents, X, y):
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
	def simulate_agents_01_loss(self, agents, X, y):
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


	def choose_the_best_classifier(self, X_train, y_train, X_val, y_val):
		agents = []
		agents.append(Agent_single_sklearn("bnb", BernoulliNB()))
		agents.append(Agent_single_sklearn("lr", LogisticRegression()))
		agents.append(Agent_single_sklearn("svc",SVC(kernel = 'poly', degree = 4, probability = True, random_state = 0)))

		#train
		for agent in agents:
			agent.clf.fit(X_train, y_train)
			if agent.clf.classes_[0] == 'Excellent':
				agent._excellent_index = 0
			else:
				agent._excellent_index = 1

		#test on validation
		value = 1000
		agent_wealths = self.simulate_agents_wealths(agents, value, X_val, y_val)
		agent_logloss = self.simulate_agents_logloss(agents, X_val, y_val)
		agent_01_loss = self.simulate_agents_01_loss(agents, X_val, y_val)

		#choose the best one
		max_wealths = 0
		select = 0
		for agent in agents:
			if agent_wealths[agent] > max_wealths:
				max_wealths = agent_wealths[agent]
				select = agent
			else:
				if agent_wealths[agent] == max_wealths:
					min_ll = 1000000
					for agent in agents:
						if agent_logloss[agent] < min_ll:
							min_ll = agent_logloss[agent]
							select = agent
						else:
							if agent_logloss[agent] == min_ll:
								min_error = 1000000
								for agent in agents:
									if agent_01_loss[agent] < min_error:
										min_error = agent_01_loss[agent]
										select = agent
		return select.clf



		
'''
simple faster method unfortunately doesn't work well T_T
'''
'''
class Agent_pguo3(Agent):
	def __init__(self, name):
		self.name = name

	def choose_the_best_classifier(self, X_train, y_train, X_val, y_val):

		agents = []
		agents.append(Agent_single_sklearn("bnb", BernoulliNB()))
		agents.append(Agent_single_sklearn("lr", LogisticRegression()))
		agents.append(Agent_single_sklearn("svc",SVC(kernel = 'poly', degree = 4, probability = True, random_state = 0)))

		num_accurate = []

		#train
		for agent in agents:
			agent.clf.fit(X_train, y_train)
			
			y_val_pred = agent.clf.predict(X_val)
			num = np.count_nonzero(y_val == y_val_pred)
			num_accurate.append(num)

		most_accurate = 0
		max_ = 0
		for i in range(len(num_accurate)):
			if num_accurate[i] >= max_:
				max_ = num_accurate[i]
				most_accurate = i
		print most_accurate
		select_agent = agents[most_accurate]
		return select_agent.clf
'''

	  
