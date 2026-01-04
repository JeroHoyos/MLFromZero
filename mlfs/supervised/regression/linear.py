import numpy as np

class LinearRegression:
    """
    Linear regression.
    
    This class implements multivariate linear regression using
    batch gradient descent. 

    Attributes
    ----------
    w : ndarray of shape (n_features,)
        Weight vector of the linear model.
    b : float
        Bias (intercept) term.
    history : list of float
        History of the loss values during training.
        
    """

    def __init__(self):
        self.w = None
        self.b = None
        self.history = []

    def fit(self, X, y, alpha=0.01, num_iters=1000):
        """
        Train the linear regression model using gradient descent.

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Training data, where m is the number of samples
            and n is the number of features.
        y : ndarray of shape (m,)
            Target values.
        alpha : float, optional
            Learning rate (default is 0.01).
        num_iters : int, optional
            Number of gradient descent iterations (default is 1000).

        """

        n = X.shape[1]
        self.w = np.zeros(n)
        self.b = 0.0

        self.gradient_descent(X, y, alpha, num_iters)
    
    def predict(self, X):
        """
        Predict output values using the trained linear model.

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (m,)
            Predicted values.

        """
        return np.dot(X, self.w) + self.b
    
    def loss(self, X, y):
        """
        Compute the Mean Squared Error (MSE) loss.

        This function measures how well the current model
        fits the training data.

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Input data.
        y : ndarray of shape (m,)
            True target values.

        Returns
        -------
        loss : float
            Mean squared error of the model predictions.

        """
        m = X.shape[0]

        loss_sum = 0.0

        for i in range(m):
            f_wb = np.dot(self.w, X[i,:]) + self.b

            loss_sum += (f_wb - y[i])**2

        return loss_sum / m
    
    def compute_gradient(self, X, y):
        """
        Compute the gradients of the loss function with respect
        to the model parameters w and b.

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Input data.
        y : ndarray of shape (m,)
            True target values.

        Returns
        -------
        dj_dw : ndarray of shape (n,)
            Gradient of the loss with respect to w.
        dj_db : float
            Gradient of the loss with respect to b.
            
        """
        m, n = X.shape

        dj_dw = np.zeros(n)
        dj_db = 0

        for i in range(m):

            f_wb = np.dot(self.w, X[i,:]) + self.b

            error = f_wb - y[i]


            dj_dw += error * X[i, :]
            dj_db += error

        dj_dw = (2 / m) * dj_dw
        dj_db = (2 / m) * dj_db
            
        return dj_dw, dj_db    

    def gradient_descent(self, X, y, alpha, num_iters):
        """
        Perform batch gradient descent to optimize the model parameters.

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Training data.
        y : ndarray of shape (m,)
            Target values.
        alpha : float
            Learning rate.
        num_iters : int
            Number of iterations.
        """
        
        for _ in range(num_iters):

            dj_dw, dj_db = self.compute_gradient(X, y)

            self.w -= alpha * dj_dw
            self.b -= alpha * dj_db

            self.history.append(self.loss(X, y))

    
