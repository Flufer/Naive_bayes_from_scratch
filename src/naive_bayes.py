import numpy as np

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
        pass

    def _log_gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, var: np.ndarray):
        """
        Логарифм плотности нормального распределения
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание классов
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание вероятностей классов
        """
        pass
    