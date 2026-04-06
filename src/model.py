import numpy as np


class LogisticRegressionGD:
    """Binary Logistic Regression optimized with Gradient Descent."""

    def __init__(self, learning_rate: float = 0.05, epochs: int = 5000, l2_lambda: float = 0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation: g(z) = 1 / (1 + exp(-z))."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary log loss: J = -(1/m) * sum(y*log(p) + (1-y)*log(1-p))."""
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train weights with gradient descent.

        Linear score: z = Xw + b
        Probability: p = sigmoid(z)
        Gradients: dw = (1/m) * X^T * (p - y) + (lambda/m) * w, db = (1/m) * sum(p - y)
        Update: w <- w - alpha * dw, b <- b - alpha * db
        """
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            linear = np.dot(x, self.weights) + self.bias
            y_pred = self._sigmoid(linear)

            dw = (1.0 / n_samples) * np.dot(x.T, (y_pred - y))
            if self.l2_lambda > 0:
                dw += (self.l2_lambda / n_samples) * self.weights
            db = (1.0 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._log_loss(y, y_pred)
            if self.l2_lambda > 0:
                loss += float((self.l2_lambda / (2.0 * n_samples)) * np.sum(self.weights ** 2))
            self.loss_history.append(loss)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class-1 probability: p_hat = sigmoid(Xw + b)."""
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        linear = np.dot(x, self.weights) + self.bias
        return self._sigmoid(linear)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convert probability to class: y_hat = 1 if p_hat >= threshold else 0."""
        probabilities = self.predict_proba(x)
        return (probabilities >= threshold).astype(int)
