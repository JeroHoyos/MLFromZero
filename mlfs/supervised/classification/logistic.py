import numpy as np
from mlfs.utils.math import sigmoid


class LogisticRegression:

    def __init__(self):
        self.w = None
        self.b = None
        self.history = []

    def fit(self, X, y, alpha=0.01, num_iters=1000):
        """
        Train the logistic regression model using gradient descent.

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Training data.
        y : ndarray of shape (m,)
            Binary target values (0 or 1).
        alpha : float, optional
            Learning rate.
        num_iters : int, optional
            Number of iterations.
        """

        n = X.shape[1]
        self.w = np.zeros(n)
        self.b = 0.0

        self.gradient_descent(X, y, alpha, num_iters)


    def predict_proba(self, X):
        """
        Predict probabilities for input data.

        Parameters
        ----------
        X : ndarray of shape (m, n)

        Returns
        -------
        probs : ndarray of shape (m,)
            Predicted probabilities.
        """

        z = np.dot(X, self.w) + self.b
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predict binary class labels.

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Input data.
        threshold : float, optional
            Classification threshold.

        Returns
        -------
        y_pred : ndarray of shape (m,)
            Predicted class labels (0 or 1).
        """

        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def loss(self, X, y):
        """
        Compute the log loss.

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
            Log loss of the model predictions.

        """

        m = X.shape[0]

        loss_sum = 0.0

        for i in range(m):
            z_wb = np.dot(self.w, X[i,:]) + self.b

            f_wb = sigmoid(z_wb)
            f_wb = np.clip(f_wb, 1e-15, 1 - 1e-15)
            
            loss = -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
            loss_sum += loss

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

            f_wb = sigmoid(np.dot(self.w, X[i, :]) + self.b)

            error = f_wb - y[i]
            dj_dw += error * X[i, :]
            dj_db += error

        dj_dw = (1 / m) * dj_dw
        dj_db = (1 / m) * dj_db
    
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