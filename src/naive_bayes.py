import numpy as np
from sklearn.covariance import log_likelihood


class GaussianNaiveBayes:
    """
    Наивный Байесовский Классификатор
    """

    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.vars = {}
        self.eps = 1e-9

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение модели
        """
        self.classes = np.unique(y)
        n_samples = X.shape[0]

        for cls in self.classes:
            X_cls = X[y == cls]

            self.priors[cls] = X_cls.shape[0] / n_samples
            self.means[cls] = X_cls.mean(axis=0)
            self.vars[cls] = X_cls.var(axis=0) + self.eps

        return self

    def _log_gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, var: np.ndarray):
        """
        Логарифм плотности нормального распределения
        """
        return -0.5 * (np.log(2 * np.pi * var) + ((X - mean) ** 2) / var).sum(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание классов
        """
        posteriors = []

        for cls in self.classes:
            log_prior = np.log(self.priors[cls])
            log_likelihood = self._log_gaussian_pdf(X, self.means[cls], self.vars[cls])
            posteriors.append(log_prior + log_likelihood)

        posteriors = np.vstack(posteriors).T
        return self.classes[np.argmax(posteriors, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание вероятностей классов
        """
        posteriors = []

        for cls in self.classes:
            log_prior = np.log(self.priors[cls])
            log_likelihood = self._log_gaussian_pdf(
                X, self.means[cls], self.vars[cls]
            )
            posteriors.append(log_prior + log_likelihood)

        log_probs = np.vstack(posteriors).T
        log_probs -= log_probs.max(axis=1, keepdims=True)

        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)
