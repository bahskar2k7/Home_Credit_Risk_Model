import pandas as pd
import sklearn.decomposition as decomp
import sklearn.ensemble as ens
import sklearn.pipeline as pipe
import sklearn.preprocessing as prep

import model.params as params
import utils

logger = utils.logger
model_tag = utils.ModelTag


def get_pipeline_random_forest() -> pipe.Pipeline:
    model = pipe.Pipeline(
        [
            ("categories", prep.OneHotEncoder(dtype=pd.CategoricalDtype, max_categories=10)),
            # ("decomposition", decomp.PCA(n_components=0.7)),
            ("clf", ens.RandomForestClassifier(**params.random_forest))
        ]
    )

    return model


def get_pipeline_random_forest_wgt() -> pipe.Pipeline:
    model = pipe.Pipeline(
        [
            # ("scaler", prep.StandardScaler()),
            # ("decomposition", decomp.PCA(n_components=0.6)),
            ("clf", ens.RandomForestClassifier(**params.random_forest_wgt))
        ]
    )

    return model
