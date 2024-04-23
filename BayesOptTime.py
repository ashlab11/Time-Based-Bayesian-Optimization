import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from . import AcquisitionFunction


def BayesOptTime(fun_to_maximize, bounds, random_state = 42, max_time = 60, initial_guesses = 5, xi=0.01):
    overall_start_time = time.time()
    random_guesses = np.random.RandomState(random_state).uniform(bounds[:, 0], bounds[:, 1], (initial_guesses, bounds.shape[0]))
    fun_values = []
    time_values = []
    
    #Go through initial guesses, before gaussian processes are fit
    for guess in random_guesses:
        start_time = time.time()
        fun_values.append(fun_to_maximize(guess))
        time_values.append(time.time() - start_time)
    gp_fun = GaussianProcessRegressor(kernel = RBF(), alpha = 1e-5, normalize_y = True)
    gp_time = GaussianProcessRegressor(kernel = RBF(), alpha = 1e-5, normalize_y = True)
    
    #Fits the gaussian processes to the time/fun values
    gp_fun.fit(random_guesses, fun_values)
    gp_time.fit(random_guesses, time_values)
    
    #Because the time integral cannot be calculated at 0, we need to choose a minimum time value
    time_min = np.min(time_values) / 10
    
    while (time.time() - overall_start_time < max_time):
        next_guess = AcquisitionFunction.maximize_eit(gp_fun, gp_time, np.max(fun_values), time_min, bounds, xi)
        random_guesses.append(next_guess)
        start_time = time.time()
        fun_values.append(fun_to_maximize(next_guess))
        time_values.append(time.time() - start_time)
        gp_fun.fit(random_guesses, fun_values)
        gp_time.fit(random_guesses, time_values)
    
    best_idx = np.argmax(fun_values)
    return random_guesses, fun_values, time_values   
    