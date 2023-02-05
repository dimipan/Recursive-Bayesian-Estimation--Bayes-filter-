"""
@author: Dimitris Panagopoulos

With that script we can heuristically choose the set of WEIGHTS variable from a list of all the possible 
values pairs could be found s.t. W1 + W2 = 1,  0 < W1,W2 < 1

We assume the best pair of WEIGHTS to be the pair that maximizes the likelihood function most of the 
times for random observations (here: angle, path). Getting the pair that maximizes the likelihood
does not mean yielding the highest posterior. That's because the latter is also affected
by other variables as well.  
"""
import numpy as np
import random
from itertools import permutations


# Define a custom key function that calculates the result of a pair of numbers
def likelihood_function(pair, a, p):
    return np.round(np.exp(-a / pair[0]) * np.exp(-p / pair[1]), 2)

def find_best_pair(Angle, Path):
    a = Angle / 180
    p = Path / 25
    # Use the max() function to find the pairs that give the maximum result
    max_pairs = [pair for pair in permutations(all_pairs, 2) if sum(pair)==1 and
                  likelihood_function(pair, a, p) == max(likelihood_function(pair, a, p) 
                  for pair in permutations(all_pairs, 2) if sum(pair)==1)]
    
    # Calculate the maximum result from the pairs
    #max_result = likelihood_function(max_pairs[0], a ,p)
    
    for pair in max_pairs:
        ls.append(pair) 
    
    return ls

def get_best(ls, iteration):
    pair_cnt = {}
    for z in ls:
        if z not in pair_cnt:
            pair_cnt[z] = 1
        else:
            pair_cnt[z] += 1
    max_cnt = max(pair_cnt.values())
    max_pairs = [y for y, count in pair_cnt.items() if count==max_cnt]
    print(f"We get {max_pairs} in a total of {max_cnt} loops")
    # check if the best pair appeared at least 40% of the iterations
    print('not sure about the yielding pair') if max_cnt < 0.45*iteration else print(' OK ')
   
    

if __name__ == "__main__": 
    all_pairs = np.round(np.arange(0.1, 1.0, 0.1).tolist(), 2)
    ls = []
    for it in range(500):
        Angle = np.array([random.randint(0, 180)])
        Path = np.array([random.randint(1, 25)])
        best_list = find_best_pair(Angle, Path)
    
    get_best(best_list, it)








    
