# TODO: SVM linear and non-linear, oneClass SVM

import pandas as pd
import sklearn.decomposition as decomp
import sklearn.pipeline as pipe
import sklearn.preprocessing as prep
import sklearn.svm as svm

import model.params as params
import utils

logger = utils.logger
model_tag = utils.ModelTag


def get_pipeline_svm_rbf() -> pipe.Pipeline:
    model = pipe.Pipeline(
        [
            ("categories", prep.OneHotEncoder(dtype=pd.CategoricalDtype, max_categories=10)),
            ("scaler", prep.StandardScaler()),
            ("decomposition", decomp.PCA(n_components=0.2)),
            ("clf", svm.SVC(**params.svm_rbf))
        ]
    )

    return model


def get_pipeline_svm_nu() -> pipe.Pipeline:
    model = pipe.Pipeline(
        [
            ("categories", prep.OneHotEncoder(dtype=pd.CategoricalDtype, max_categories=10)),
            ("scaler", prep.StandardScaler()),
            ("decomposition", decomp.PCA(n_components=0.2)),
            ("clf", svm.NuSVC(**params.svm_nu))
        ]
    )

    return model


def get_pipeline_svm_linear() -> pipe.Pipeline:
    model = pipe.Pipeline(
        [
            ("categories", prep.OneHotEncoder(dtype=pd.CategoricalDtype, max_categories=10)),
            ("scaler", prep.StandardScaler()),
            ("decomposition", decomp.PCA(n_components=0.2)),
            ("clf", svm.SVC(**params.svm_linear))
        ]
    )

    return model
