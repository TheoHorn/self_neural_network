import numpy as np

class Error_functions:
    def quadratic_error(outputs, targets):
        return 0.5 * np.sum(np.square(outputs - targets))

    def mean_squared_error(outputs, targets):
        return np.mean(np.square(outputs - targets))
    
    def cross_entropy_error(outputs, targets):
        return -np.sum(targets * np.log(outputs))
    
    def mean_absolute_error(outputs, targets):
        return np.mean(np.abs(outputs - targets))
    
    def derivate_mean_squared_error(outputs, targets):
        return outputs - targets
    
    def derivate_cross_entropy_error(outputs, targets):
        return outputs - targets
    
    def derivate_mean_absolute_error(outputs, targets):
        return np.sign(outputs - targets)
    
    def derivate_quadradic_error(outputs, targets):
        return outputs - targets
    
    def get_value_from_name(outputs, targets, name):
        if name == "mean_squared_error":
            return Error_Functions.mean_squared_error(outputs, targets)
        elif name == "cross_entropy_error":
            return Error_Functions.cross_entropy_error(outputs, targets)
        elif name == "mean_absolute_error":
            return Error_Functions.mean_absolute_error(outputs, targets)
        else:
            raise ValueError("Error function not supported")