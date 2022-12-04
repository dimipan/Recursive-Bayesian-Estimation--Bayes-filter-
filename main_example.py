"""
Copyright (c) 2022-2023
All rights reserved.
@author: Dimitris Panagopoulos

The following example is given to understand the utility and requirements of the algorithm to get it working. We assume 
that a robot agent navigates around and there are three hidden variables representing three goals arranged in a cartesian 
coordinate system. The number of total observation sources (i.e., the sensors that data is collected from) are two.
For better intuition, the two sources are robot's angle and distance w.r.t each hidden variable (i.e., goal). We define
two normalization values, each one corresponding to the maximum measurement value that source can get. Next, the 
weights for each observation source are chosen. Finally, we choose a Delta value based on our preference on how much we 
want the current transition state model to be dependent on the previous one.

The most important factor is how the observation values should be inserted in the script. The values should be all included
in one list in the following format. If we have 3 hidden variables and 2 observation sources, then the length of the list 
should be 6. The first two positions of the list stand for the first hidden variable with index 0 to be its angle value and
index 1 the distance value. The next two positions account for the second hidden variable with the index 2 to be its angle
value and index 3 to be its distance value, and so on for the last hidden variable capturing the last two indices in the list.     
   
""" 
import random
from Bayesian_Estimation import RecursiveBayesianEstimation 

if __name__ == "__main__":
    
    NUM_HIDDEN_VARIABLES = 3  # set the number of hidden variables to be estimated
    NUM_TOTAL_OBSERVATIONS = 2 # set the number of observation sources
    NORMALIZATION_VALUES = [180, 25] # length should be equal to total observations
    WEIGHTS = [0.6, 0.4] # length should be equal to total observations
    DELTA = 0.2
    
    bayes = RecursiveBayesianEstimation(NUM_HIDDEN_VARIABLES, NUM_TOTAL_OBSERVATIONS, NORMALIZATION_VALUES, WEIGHTS, DELTA)
    
    ITERATOR = 1
    EPISODES = 2
    while ITERATOR <= EPISODES:
        print('Iteration No :', ITERATOR)
        # dummy example for demonstration purposes
        observations = [random.randint(0, 180), random.randint(1, 25), random.randint(0, 180), random.randint(1, 25), random.randint(0, 180), random.randint(1, 25)]
        
        print(observations)
        bayes.get_bayesian_update(observations)
        ITERATOR += 1
    