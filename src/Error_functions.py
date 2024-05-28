import numpy as np

class Error_functions:
    def quadratic_error(outputs, targets):
        return 0.5 * np.sum(np.square(outputs - targets))

    def mean_squared_error(outputs, targets):
        return np.mean(np.square(outputs - targets))

    def cross_entropy_error(outputs, targets):
        # Add a small value to avoid log(0)
        return -np.sum(targets * np.log(outputs + 1e-9))

    def mean_absolute_error(outputs, targets):
        return np.mean(np.abs(outputs - targets))

    def derivate_mean_squared_error(outputs, targets):
        return 2 * (outputs - targets) / targets.size

    def derivate_cross_entropy_error(outputs, targets):
        # Add a small value to avoid division by zero
        return -(targets / (outputs + 1e-9))

    def derivate_mean_absolute_error(outputs, targets):
        # Use np.where to handle the case where outputs - targets == 0
        return np.where(outputs - targets > 0, 1, np.where(outputs - targets < 0, -1, 0)) / targets.size

    def derivate_quadradic_error(outputs, targets):
        return outputs - targets

    def get_value_from_name(outputs, targets, name):
        if name == "mean_squared_error":
            return Error_functions.mean_squared_error(outputs, targets)
        elif name == "cross_entropy_error":
            return Error_functions.cross_entropy_error(outputs, targets)
        elif name == "mean_absolute_error":
            return Error_functions.mean_absolute_error(outputs, targets)
        elif name == "quadratic_error":
            return Error_functions.quadratic_error(outputs, targets)
        else:
            raise ValueError("Error function not supported")
