import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from AcquisitionFunction import maximize_eit, eit 


def BayesOptTime(fun_to_maximize, bounds, max_time = 60, initial_guesses = 5, xi=0.01):
    bounds = np.array(bounds)
    
    overall_start_time = time.time()
    random_guesses = np.random.uniform(bounds[:, 0], bounds[:, 1], (initial_guesses, bounds.shape[0]))
    fun_values = []
    time_values = []
    
    max_val = -np.inf
    max_vals = []
    #Go through initial guesses, before gaussian processes are fit
    for guess in random_guesses:
        start_time = time.time()
        fun_value = fun_to_maximize(*guess)
        fun_values.append(fun_value)
        time_values.append(time.time() - start_time)
        if fun_value > max_val:
            max_val = fun_value
            max_vals.append(max_val)
            
    print("Maximum value found prior to optimization: ", max_val)
    gp_fun = GaussianProcessRegressor(kernel = Matern(nu = 2.5), alpha = 1e-6, normalize_y = True, n_restarts_optimizer = 5)
    gp_time = GaussianProcessRegressor(kernel = RBF(), alpha = 1e-6, normalize_y = True, n_restarts_optimizer = 5)
    
   #Fits the gaussian processes to the time/fun values
    gp_fun.fit(random_guesses, fun_values)
    gp_time.fit(random_guesses, time_values)
    
    #Because the time integral cannot be calculated at 0, we need to choose a minimum time value
    time_min = np.min(time_values) / 2
    
    while (time.time() - overall_start_time < max_time):
        next_guess = maximize_eit(gp_fun, gp_time, np.max(fun_values), time_min, bounds, xi)
        random_guesses = np.append(random_guesses, next_guess.reshape(1, -1), axis = 0)
        start_time = time.time()
        fun_value = fun_to_maximize(*next_guess)
        fun_values.append(fun_value)
        if fun_value > max_val:
            max_val = fun_value
            max_vals.append(max_val)
            print("Maximum value found: ", max_val)
        time_values.append(time.time() - start_time)
        gp_fun.fit(random_guesses, fun_values)
        gp_time.fit(random_guesses, time_values)
    
    best_idx = np.argmax(fun_values)
    return random_guesses, fun_values, time_values   
    