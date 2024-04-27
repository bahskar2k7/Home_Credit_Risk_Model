import typing as t
from pathlib import Path
import joblib


class ModelTag:
    """
    Wrapper class to load/save model, its params and results.

    https://stackoverflow.com/questions/56107259/how-to-save-a-trained-model-by-scikit-learn
    https://scikit-learn.org/stable/model_persistence.html

    """

    def __init__(
            self,
            model: t.Any,
            model_name: str,
            results: dict,
            model_path: t.Union[str, Path]
    ):
        """

        :param model: model object
        :param model_name: name of model
        :param results: dictionary containing scores
        :param model_path: directory into which to save model and its results
        """
        self.model = model
        self.model_name = model_name
        self.results = results
        self.model_path = model_path

    def load(self):
        """
        Load model
        :return:
        """
        pass

    def save(self):
        """
        Save model using joblib
        :return:
        """
        model = self.model
        save_path = self.model_path
        results = self.results
        save_path.mkdir(exist_ok=True)

        joblib.dump(model, save_path.joinpath("model.pkl"))

        # write dict to text file in same folder
        with open(save_path.joinpath("results.txt"), "w+") as fl:
            fl.write(str(results))

    def get_results(self):
        pass
