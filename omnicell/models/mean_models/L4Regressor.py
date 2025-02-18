from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
from sklearn.exceptions import NotFittedError
import numpy as np
class L4Regressor(BaseEstimator, RegressorMixin):
    """Custom regressor using L4 loss for multi-output regression."""
    
    def __init__(self, max_iter=1000, tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None  # Coefficients matrix (n_features + 1, n_outputs)

    def fit(self, X, Y):
        # Add intercept term
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Initialize coefficients matrix (including intercept)
        n_features = X_with_intercept.shape[1]
        n_outputs = Y.shape[1]
        initial_coef = np.zeros((n_features, n_outputs))
        
        # Flatten for optimization
        initial_coef_flat = initial_coef.ravel()
        
        # Define loss for multi-output
        def loss(coef_flat):
            coef = coef_flat.reshape(n_features, n_outputs)
            residuals = Y - X_with_intercept @ coef
            return np.sum(residuals**4)
        
        # Optimize
        result = minimize(loss, initial_coef_flat, method='L-BFGS-B',
                          options={'maxiter': self.max_iter, 'gtol': self.tol})
        
        self.coef_ = result.x.reshape(n_features, n_outputs)
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise NotFittedError("Model not fitted yet.")
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_with_intercept @ self.coef_