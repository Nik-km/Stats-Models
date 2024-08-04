import scipy, math
import numpy as np
x = math.factorial(7)

#>> Monte Carlo Simulation
def monte_carlo(n):
    count = 0       # to count the # of favourable cases
    for _ in range(n):
        arr = np.random.uniform(0,2,3)  # gen 3 #'s under uniform distribution over [0,2]
        med = np.median(arr)    # find median above 3 #'s
        if med > 1.5:           # satisfying the condition
            count += 1
    return count/n  # return the probability

# Choosing an arbitrarily large number for better approximation
monte_carlo(10**6)
