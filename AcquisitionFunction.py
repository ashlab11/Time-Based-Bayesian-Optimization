from scipy.integrate import quad
from scipy.stats import norm
import numpy as np

def maximize_eit(gp_fun, gp_time, fun_max, bounds, xi=0.01):
    pass

def eit(x, gp_fun, gp_time, fun_max, xi=0.01):
    "Calculates the expected improvement over time acquisition function at a value x"
    mean_fun, std_fun = gp_fun.predict(x, return_std=True)
    mean_time, std_time = gp_time.predict(x, return_std=True)
    
    #First, calculate the expected improvement at x
    a = mean_fun - fun_max - xi
    z = a / std_fun
    ei_fun = a * norm.cdf(z) + std_fun * norm.pdf(z)
    
    #Then, calculate the expected inverse time at x
    min_time = mean_time - 3 * std_time
    
    integrand = lambda y: (1/(y * std_time * np.sqrt(2 * np.pi))) * np.exp(-((y - mean_time)**2) / (2 * std_time**2))
    expected_inverse_time, _ = quad(integrand, min_time, np.inf)
    
    return ei_fun / expected_inverse_time
    
    