"""
Copyright (c) 2022-2023
All rights reserved.
@author: Dimitris Panagopoulos

Based on the algorithm described in the paper "A Bayesian-based approach to Human-Operator Intent Recognition in Remote Mobile Robot
Navigation" that was published in the SMC 2021, this script is a general modular (plug-n-play) version of the algorithm for any
problem that utilizes recursive Bayesian estimation.
The algorithm needs:
    - the number of hidden variables 'x'
    - the number of total observation sources used to get measurements
    - the normalization values (i.e., maximum values that each observation source gets)
    - the weights that account for the development of the observation model according to the paper
    - the constant Delta(Δ) that defines the conditional probability table according to paper
    - the observation values in list (see 'main.py' for extra info and correct format)

"""

import numpy as np

class RecursiveBayesianEstimation:
    def __init__(self, hidden_variables, total_observations, normalization_value, weights, Delta):
        self.hidden_variables = hidden_variables       # the number of hidden variables to be estimated
        self.total_observations = total_observations   # the number of observation sources
        self.normalization_value = normalization_value # length should be equal to total observations
        self.weights = weights                         # length should be equal to total observations
        self.Delta = Delta                             # defines the conditional probability table according to paper
        
        self.prior = np.ones(self.hidden_variables) * 1/self.hidden_variables  # uniform distribution at first step
        
        conditional_probability = np.ones((self.hidden_variables, self.hidden_variables)) * (self.Delta / (self.hidden_variables-1))
        np.fill_diagonal(conditional_probability, 1-self.Delta)  # EQUATION 3 
        
        decimal_places = 2          # auxiliary constant for rounding results
        reset_array = np.array([])  # init empty array for storing data 
        self.constants = [conditional_probability, decimal_places, reset_array]  # set constants needed for the algorithm
        
        
    def get_observation_model(self, observations):  # Π [Pr(z{t} | x{t})] - EQUATION 1 (first term) 
        values_obs_model = self.constants[2]  # init values_obs array for storing observation data
        for idx in range(len(observations)):
            idx += 1
            if idx % self.total_observations == 0:  # do calculations every TOTAL OBSERVATIONS number
                for z in range(self.total_observations):
                    idx_value = observations[idx-self.total_observations+z] / self.normalization_value[z]
                    values_obs_model = np.append(values_obs_model, idx_value) 
        observation_output = self.constants[2]
        for i in range(self.hidden_variables):
            paired_values = values_obs_model[0:self.total_observations]
            store = self.constants[2]
            for index in range(self.total_observations):
                exp_model = np.exp(-paired_values[index] / self.weights[index])
                store = np.append(store, exp_model)
            observation_output = np.append(observation_output, np.prod(store))
            values_obs_model = values_obs_model[self.total_observations:]
        return observation_output
    
  
    def get_transition_model(self):  # Pr(x{t} | x{t-1}) * Pr(x{t-1}) - EQUATION 1 (second term)
        print("prior = ", self.get_round_result(self.prior))
        return np.matmul(self.constants[0], self.get_round_result(self.prior.T))
        
    
    def get_posterior_belief(self, observation_output, transition_output):  # Pr(x{t} | z{1:t}) - EQUATION 1
        return observation_output * transition_output / np.sum(observation_output * transition_output)
    
    
    def get_maximum_and_recursion(self, normalized_posterior_belief): # EQUATION 2
        self.prior = normalized_posterior_belief
        if not np.sum(normalized_posterior_belief) == 1:
            raise Exception(""""By definition, it is not possible for the posterior probability to exceed one
                            Might be corrupted by decimals. Try and change decimal places constant""")
        return np.argmax(normalized_posterior_belief)
    
    
    def get_round_result(self, element): # auxiliary function to reduce decimal places
        return np.round(element, self.constants[1])
        
    
    def get_bayesian_update(self, observations):  # here the estimation is performed
        if not self.total_observations == len(self.weights) == len(self.normalization_value) and \
            not self.hidden_variables*self.total_observations == len(observations):
                raise Exception("""Sorry, estimation cannot be executed. You have to check dimensions first.
                            Make sure that: num_total_observations = len(weights) = len(normalization_values) AND
                            len(observations) = num_hidden_variables * num_total_observations""")
        likelihood = self.get_round_result(self.get_observation_model(observations))
        transition = self.get_round_result(self.get_transition_model())
        posterior = self.get_round_result(self.get_posterior_belief(likelihood, transition))
        estimated_variable = self.get_maximum_and_recursion(posterior)
        print("likelihood = ", likelihood)
        print("transition model = ", transition)
        print("posterior = ", posterior)
        print("estimated variable X =", estimated_variable+1)
        print("--------------------")
        
        return estimated_variable
        