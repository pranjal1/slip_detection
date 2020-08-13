from abc import ABC, abstractmethod


class PredictionModelInterface(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        super().__init__()

    @abstractmethod
    def preprocess_video(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass
