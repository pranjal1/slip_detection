from loguru import logger
from .model_interface import PredictionModelInterface


class FallDetect(PredictionModelInterface):
    """
    Model for Fall-Detection-with-CNNs-and-Optical-Flow
    https://github.com/AdrianNunez/Fall-Detection-with-CNNs-and-Optical-Flow
    """

    def __init__(self, model_path):
        super().__init__(model_path)
        self.model_path = model_path
        self.model = None

    def preprocess_video(self, video_path):
        logger.info(f"Processing video in path {video_path}...")
        video_processor = lambda x: x
        return video_processor(video_path)

    def load_model(self):
        model_loader = lambda x: x
        logger.info(f"Loading model in path {self.model_path}...")
        self.model = model_loader(self.model_path)
        logger.info(f"Loading model done.")

    def predict(self, video_path):
        video = self.preprocess_video(video_path)
        # ideally the command below should have been executed
        logger.info(f"Predicting...")
        # return self.model(video)
        logger.info(f"Prediction generated!")
        # time period where fall event has occured
        return [(93, 102), (134, 140)]


if __name__ == "__main__":
    f = FallDetect("no_model_rn")
    f.load_model()
    print(f.predict("video_path"))
