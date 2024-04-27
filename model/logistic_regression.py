import typing as t

import pandas as pd
import sklearn.decomposition as decomp
import sklearn.linear_model as lm
import sklearn.pipeline as pipe
import sklearn.preprocessing as prep

import data
import evaluate as evl
import model.params as params
import settings as s
import utils

logger = utils.logger
model_tag = utils.ModelTag


def get_pipeline_logistic_regression() -> pipe.Pipeline:
    model = pipe.Pipeline(
        [
            ("scaler", prep.RobustScaler()),
            ("decomposition", decomp.PCA(n_components=0.7)),
            # ("decomposition", decomp.TruncatedSVD(n_components=0.6)),
            ("clf", lm.LogisticRegression(**params.lg_l2_c))
        ]
    )

    return model


def get_pipeline_logistic_regression_l2() -> pipe.Pipeline:
    model = pipe.Pipeline(
        [
            ("scaler", prep.StandardScaler()),
            ("decomposition", decomp.PCA(n_components=0.6)),
            ("clf", lm.LogisticRegression(**params.lg_l2))
        ]
    )

    return model


def get_pipeline_logistic_regression_catg() -> pipe.Pipeline:
    model = pipe.Pipeline(
        [
            # ("categories", prep.OneHotEncoder(dtype=pd.CategoricalDtype, max_categories=10)),
            ("scaler", prep.StandardScaler()),
            ("decomposition", decomp.PCA(n_components=0.6)),
            ("clf", lm.LogisticRegression(**params.lg_l2))
        ]
    )

    return model


def run(model_name: str, pipeline_callable: t.Callable):
    # https://towardsdatascience.com/a-brief-introduction-to-scikit-pipelines-888edc86da9b
    df = data.load_data()

    # need numeric variables
    df = df.select_dtypes(exclude=['category', 'datetime64[ns]'])

    x_train, y_train, x_test, y_test = data.split_data(df, 0.7)

    model: pipe.Pipeline = pipeline_callable()

    logger.info("Start training model")
    model.fit(x_train, y_train)

    logger.info("Making predictions")
    y_pred = model.predict(x_test)

    scores = evl.evaluate(y_test, y_pred)
    model_params = model.get_params()

    # save model and results
    save_path = s.PATH.MODEL_OUT.joinpath(f"{model_name}_{s.get_timestamp_str()}")
    save_path.mkdir(exist_ok=True)
    model_results = {
        'model': model_name,
        'path': str(save_path),
        'params': model_params,
        'scores': scores

    }
    tag = model_tag(
        model=model,
        model_name=model_name,
        results=model_results,
        model_path=save_path
    )
    tag.save()


if __name__ == "__main__":
    run("logistic-regression", get_pipeline_logistic_regression_catg)
