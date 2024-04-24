from scipy.integrate import quad
from scipy.optimize import minimize, Bounds
from scipy.stats import norm
import numpy as np

def maximize_eit(gp_fun, gp_time, fun_max, time_min, bounds, xi=0.01) -> np.ndarray:
    """
    Maximizes the expected improvement over time acquisition function. Parameters:
    gp_fun: sklearn GaussianProcessRegressor, the surrogate model for the function
    gp_time: sklearn GaussianProcessRegressor, the surrogate model for the time
    fun_max: float, the maximum value of the function
    time_min: float, the minimum time value to begin integration at
    bounds: np.ndarray, the bounds of the function
    xi: float, the exploration-exploitation tradeoff parameter"""
    
    #Need to minimize eit, not maximize it -- so create a negative lambda function on eit
    eit_neg = lambda x: -eit(x, gp_fun, gp_time, fun_max, time_min, xi)
    optimizing_bounds = Bounds(bounds[:, 0], bounds[:, 1])
    min_x = minimize(eit_neg, args = (gp_fun, gp_time, fun_max, time_min), bounds = optimizing_bounds)
    return min_x

def eit(x, gp_fun, gp_time, fun_max, time_min, xi=0.01) -> float:
    """
    Calculates the expected improvement over time acquisition function at a value x. Parameters:
    
    x: float, the value at which to calculate the acquisition function
    
    gp_fun: skleardn GaussianProcessRegressor, the surrogate model for the function
    
    gp_time: sklearn GaussianProcessRegressor, the surrogate model for the time
    
    fun_max: float, the maximum value of the function
    
    time_min: float, the minimum time value to begin integration at
    
    xi: float, the exploration-exploitation tradeoff parameter"""
    
    mean_fun, std_fun = gp_fun.predict(x, return_std=True)
    mean_time, std_time = gp_time.predict(x, return_std=True)
    
    #First, calculate the expected improvement at x
    a = mean_fun - fun_max - xi
    z = a / std_fun
    ei_fun = a * norm.cdf(z) + std_fun * norm.pdf(z)
        
    integrand = lambda y: (1/(y * std_time * np.sqrt(2 * np.pi))) * np.exp(-((y - mean_time)**2) / (2 * std_time**2))
    expected_inverse_time, _ = quad(integrand, time_min, np.inf)
    
    return ei_fun / expected_inverse_time
    
    